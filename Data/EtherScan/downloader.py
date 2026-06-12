import os
import json
import sys
import argparse
from pathlib import Path
from threading import Lock, Semaphore
import time
import logging
from tqdm import tqdm
import pandas as pd
import re
import requests
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ContractsDownloader:
    def __init__(self, addr_csv, api_csv, output_dir, chain_id, threads):
        if not Path(addr_csv).is_file():
            raise FileNotFoundError(f"Addresses CSV {addr_csv} does not exist")
        if not Path(api_csv).is_file():
            raise FileNotFoundError(f"API keys CSV {api_csv} does not exist")
        if threads < 1:
            raise ValueError("Threads must be positive")
        if chain_id not in [1, 137, 56]:  # Add more supported chain IDs as needed
            raise ValueError(f"Unsupported chain_id: {chain_id}")
        self.addr_csv = addr_csv
        self.api_csv = api_csv
        self.output_dir = output_dir
        self.chain_id = chain_id
        self.threads = threads
        self.tokens = []
        self.prag_re = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
        self.lock = Lock()
        self.load_tokens()

    def load_existing_metadata(self, metadata_path="contracts_metadata.csv"):
        """Load existing metadata and return set of already downloaded addresses and rows."""
        if os.path.exists(metadata_path):
            try:
                df = pd.read_csv(metadata_path)
                done = set(df["address"].astype(str))
                rows = df.values.tolist()
                return done, rows
            except Exception as e:
                logging.warning(f"Could not load existing metadata: {e}")
        return set(), []

    def load_tokens(self):
        df = pd.read_csv(self.api_csv)
        self.tokens = [str(x) for x in df.iloc[:, 0].values]
        self.threads = min(self.threads, len(self.tokens))
        logging.info(
            f"Loaded {len(self.tokens)} API keys, using {self.threads} threads"
        )

    def test_api_keys(self):
        """Test all API keys with a known contract."""
        test_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # USDT
        valid = []
        invalid = []
        print(self.tokens)
        for i, token in enumerate(self.tokens):
            try:
                sourcecode = self.download_contract(test_address, token)
                print(
                    f"Testing key {i+1}/{len(self.tokens)}: {token[:6]}...{token[-4:]}"
                )
                if sourcecode and len(sourcecode) > 0:
                    valid.append(token)
                else:
                    invalid.append(token)
                    logging.warning(
                        f"API key {i+1} failed: {token[:6]}...{token[-4:]} | No valid response"
                    )
            except Exception as e:
                logging.warning(
                    f"API key {i+1} failed: {token[:6]}...{token[-4:]} | {e}"
                )
                invalid.append(token)
        logging.info(f"API key test: {len(valid)} valid, {len(invalid)} invalid")
        if not valid:
            logging.error("No valid API keys found")
            return False
        return True

    def download_contract(self, address, token):
        """Download contract source code using Etherscan API."""
        url = "https://api.etherscan.io/v2/api/"
        params = {
            "chainid": self.chain_id,
            "module": "contract",
            "action": "getsourcecode",
            "address": address,
            "apikey": token,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "1" and data.get("result"):
                return data["result"]
            else:
                logging.warning(
                    f"API error for {address}: {data.get('message', 'Unknown')}"
                )
                return []
        except requests.RequestException as e:
            logging.error(f"Network error for {address}: {e}")
            return []

    def _pre_scan_and_sort(self):
        """Sort addresses by tx_count (descending) using pandas."""
        df = pd.read_csv(self.addr_csv)
        if df.shape[1] == 1:
            df.columns = ["address"]
            df["tx_count"] = 0
        else:
            df.columns = ["address", "tx_count"]
        df["address"] = df["address"].astype(str).str.strip()
        df["tx_count"] = (
            pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(int)
        )
        df_sorted = df.sort_values(by="tx_count", ascending=False)
        out_path = Path("addresses_sorted_by_txcount.csv")
        df_sorted.to_csv(out_path, index=False)
        return df_sorted["address"].tolist()

    def _extract_metadata(self, address, contract_path, sourcecode_obj):
        """Extract metadata from contract source code."""
        try:
            tx_count = sourcecode_obj.get(
                "TransactionCount", sourcecode_obj.get("tx_count", 0)
            )
            balance = sourcecode_obj.get("Balance", 0)
            contract_creator = sourcecode_obj.get("ContractCreator", "")
            bytecode = sourcecode_obj.get(
                "Bytecode", sourcecode_obj.get("SourceCode", "")
            )
            optimization_used = sourcecode_obj.get("OptimizationUsed", "")
            runs = sourcecode_obj.get("Runs", "")
            compiler_version = sourcecode_obj.get("CompilerVersion", "")
            license_type = sourcecode_obj.get("LicenseType", "")
            contract_name = sourcecode_obj.get("ContractName", "")

            pragma = ""
            src = ""
            if isinstance(sourcecode_obj.get("SourceCode"), str):
                src = sourcecode_obj.get("SourceCode")
            elif isinstance(sourcecode_obj.get("SourceCode"), dict):
                src = json.dumps(sourcecode_obj.get("SourceCode"))
            elif isinstance(sourcecode_obj.get("SourceCode"), list):
                src = "\n".join(
                    [c.get("SourceCode", "") for c in sourcecode_obj.get("SourceCode")]
                )

            # Skip if ABI contains 'Contract source code not verified'
            abi = sourcecode_obj.get("ABI", "")
            if isinstance(abi, str) and "Contract source code not verified" in abi:
                logging.info(
                    f"Skipped {address} due to unverified contract"
                )
                return None

            m = self.prag_re.search(src)
            if m:
                pragma = m.group(1).strip()
                ver_match = re.search(r"(\d+)\.(\d+)\.(\d+)|(\d+)\.(\d+)", pragma)
                if ver_match:
                    groups = [g for g in ver_match.groups() if g is not None]
                    if len(groups) >= 2 and int(groups[0]) == 0 and int(groups[1]) <= 3:
                        return None

            similar_bytecode = str(hash(bytecode))[:16] if bytecode else ""
            return [
                address,
                str(contract_path),
                int(tx_count),
                int(balance),
                pragma,
                contract_creator,
                similar_bytecode,
                bytecode,
                optimization_used,
                runs,
                compiler_version,
                license_type,
                contract_name,
            ]
        except Exception as e:
            logging.error(f"Metadata extraction failed for {address}: {e}")
            return None

    def download_worker(self, addresses, token, position):
        """Worker function for downloading contracts in a thread. Saves metadata every 10 contracts."""
        import pandas as pd

        metadata_rows = []
        not_valid = []
        pbar = tqdm(
            total=len(addresses),
            position=position,
            desc=f"Thread {position + 1}/{self.threads}",
        )

        def save_metadata_chunk(rows):
            if not rows:
                return
            columns = [
                "address",
                "file_path",
                "tx_count",
                "balance",
                "pragma",
                "contract_creator",
                "similar_bytecode",
                "bytecode",
                "optimization_used",
                "runs",
                "compiler_version",
                "license_type",
                "contract_name",
            ]
            df = pd.DataFrame(rows, columns=columns)
            with self.lock:
                if os.path.exists("contracts_metadata.csv"):
                    df_existing = pd.read_csv("contracts_metadata.csv")
                    df = pd.concat(
                        [df_existing, df], ignore_index=True
                    ).drop_duplicates(subset=["address"])
                df.to_csv("contracts_metadata.csv", index=False)
            logging.info(f"Thread {position+1}: Saved {len(rows)} new metadata rows.")

        for count, address in enumerate(addresses, 1):
            if address in not_valid:
                continue
            if not re.match(r"^0x[a-fA-F0-9]{40}$", address):
                logging.warning(f"Invalid address format: {address}")
                with self.lock:
                    not_valid.append(address)
                continue

            pbar.update(1)
            contract_path = Path(self.output_dir, address + ".json")
            meta = {
                "index": f"{count}/{len(addresses)}",
                "token": f"{token[:6]}...{token[-4:]}",
            }
            if os.path.exists(contract_path):
                pbar.set_postfix(meta)
                continue

            try:
                sourcecode = self.download_contract(address, token)
                if not sourcecode or len(sourcecode) == 0:
                    logging.warning(f"No source code for {address}")
                    with self.lock:
                        not_valid.append(address)
                    continue

                md = self._extract_metadata(address, contract_path, sourcecode[0])
                if md is None:
                    logging.info(f"Skipped {address} due to too old pragma (<=0.3.x)")
                    continue

                with open(contract_path, "w") as fd:
                    json.dump(sourcecode[0], fd)
                metadata_rows.append(md)

                # Save every 10 contracts
                if len(metadata_rows) >= 10:
                    save_metadata_chunk(metadata_rows)
                    metadata_rows = []

            except Exception as e:
                logging.error(f"Error processing {address}: {e}")
                with self.lock:
                    not_valid.append(address)
            finally:
                pbar.set_postfix(meta)
                time.sleep(0.2)  # Rate limit: 5 reqs/sec per key

        # Save any remaining metadata
        if metadata_rows:
            save_metadata_chunk(metadata_rows)

        return [], not_valid

    def download(self):
        """Main download function with threading."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        not_valid = []
        if os.path.exists("not_valid.json"):
            with self.lock:
                with open("not_valid.json") as fd:
                    not_valid = json.load(fd)

        addresses = self._pre_scan_and_sort()

        # Load checkpoint metadata
        done_addrs, all_metadata = self.load_existing_metadata()
        # Continue from the last done address, preserving order
        if done_addrs:
            last_done = None
            # Find the last address in the original order that is in done_addrs
            for addr in reversed(addresses):
                if addr in done_addrs:
                    last_done = addr
                    break
            if last_done:
                idx = addresses.index(last_done)
                addresses = addresses[idx+1:]

        # Distribute addresses across threads
        chunk_size = max(1, len(addresses) // self.threads)
        address_chunks = [
            addresses[i : i + chunk_size] for i in range(0, len(addresses), chunk_size)
        ]
        if len(address_chunks) > self.threads:
            address_chunks = address_chunks[: self.threads]  # Trim excess

        # Assign tokens to threads (rotate through keys)
        token_cycle = [self.tokens[i % len(self.tokens)] for i in range(self.threads)]

        all_not_valid = not_valid
        try:
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                sem = Semaphore(self.threads)
                futures = []
                for i, (chunk, token) in enumerate(zip(address_chunks, token_cycle)):
                    with sem:
                        futures.append(
                            executor.submit(self.download_worker, chunk, token, i)
                        )

                for future in futures:
                    metadata, invalid = future.result()
                    all_metadata.extend(metadata)
                    all_not_valid.extend(invalid)
        except KeyboardInterrupt:
            logging.warning("Interrupted by user. Saving progress and exiting...")

        # Save metadata (append or write)
        if all_metadata:
            import pandas as pd

            columns = [
                "address",
                "file_path",
                "tx_count",
                "balance",
                "pragma",
                "contract_creator",
                "similar_bytecode",
                "bytecode",
                "optimization_used",
                "runs",
                "compiler_version",
                "license_type",
                "contract_name",
            ]
            df = pd.DataFrame(all_metadata, columns=columns)
            if os.path.exists("contracts_metadata.csv"):
                df_existing = pd.read_csv("contracts_metadata.csv")
                df = pd.concat([df_existing, df], ignore_index=True).drop_duplicates(
                    subset=["address"]
                )
            df.to_csv("contracts_metadata.csv", index=False)
            logging.info(
                f"Saved metadata for {len(df)} contracts to contracts_metadata.csv"
            )

        # Save invalid addresses
        if all_not_valid:
            with open("not_valid.json", "w") as fd:
                json.dump(list(set(all_not_valid)), fd)
            logging.info(
                f"Saved {len(set(all_not_valid))} invalid addresses to not_valid.json"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Ethereum contracts from Etherscan."
    )
    parser.add_argument(
        "--addr-csv",
        type=str,
        default="all_contracts.csv",
        help="CSV file with contract addresses",
    )
    parser.add_argument(
        "--api-csv",
        type=str,
        default="apis.csv",
        help="CSV file with Etherscan API keys",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for contract JSONs",
    )
    parser.add_argument(
        "--chain-id",
        type=int,
        default=1,
        help="Chain ID (e.g., 1 for Ethereum Mainnet)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of concurrent threads"
    )
    args = parser.parse_args()

    downloader = ContractsDownloader(
        addr_csv=args.addr_csv,
        api_csv=args.api_csv,
        output_dir=args.output_dir,
        chain_id=args.chain_id,
        threads=args.threads,
    )
    if not downloader.test_api_keys():
        sys.exit(1)
    downloader.download()

$directories = @(
    ".\DAppSCAN-source",
    ".\DAppSCAN-bytecode"
)

foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Write-Host "Processing directory: $dir"
        
        # Get all items (folders and files) recursively
        Get-ChildItem -Path $dir -Recurse | ForEach-Object {
            $path = $_.FullName
            if ($path -like "* *") {
                $newPath = $path -replace " ", "_"
                
                # Ensure parent directory exists (in case we're renaming nested items)
                $parentDir = Split-Path -Parent $newPath
                if (!(Test-Path $parentDir)) {
                    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
                }
                
                # Rename the item
                Write-Host "Renaming: $path -> $newPath"
                Move-Item -Path $path -Destination $newPath -Force
            }
        }
    } else {
        Write-Warning "Directory not found: $dir"
    }
}

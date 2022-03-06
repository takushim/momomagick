# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [Switch]$gpu = $false,
      [Switch]$invert = $false,
      [Int]$radius = 4,
      [Float]$scale = 2.0,
      [String[]]$images)

if ($help -eq $true) {
      Write-Host ("Example: {0} -images IMAGE_LIST" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
. $HOME\.venv\gpu\Scripts\Activate.ps1
$scriptpath = "$HOME\bin\dispim"
$script = [IO.Path]::Combine($scriptpath, "mmmark.py")
$record_suffix = "_track.json"
$globs = @("*_8bit.tif", "*.tif")

# prepare the image list
if ($images.Count -eq 0){
      foreach ($glob in $globs) {
            if (Test-Path $glob) {
                  $images = (get-item $glob_path)
                  break
            }
      }
      if ($images.Count -eq 0) {
            Write-Host ("Image not found.")
      }
}
else {
      $images = (get-item $images)
}

# run
foreach ($image in $images) {
      Write-Host ("***** {0} *****" -f $image)

      $record_file = (get-item $image).BaseName.Replace("_marked", "").Replace("_8bit", "") + $record_suffix

      if ((Test-Path $record_file) -eq $false) {
            Write-Host ("Cannot find a record file: {0}" -f $record_file)
            continue
      }
      else {
            Write-Host ("Found a record file: {0}." -f $record_file)
      }

      $arglist = @($script, "-x", $scale, "-r", $radius, "-f", $record_file)

      if ($gpu -eq $true) {
            $arglist = $arglist + @("-g", "0")
      }

      if ($invert -eq $true) {
            $arglist = $arglist + @("-i")
      }

      $parameters = @{
            FilePath = (Get-command "py.exe")
            ArgumentList = $arglist + $image
      }

      Start-Process -NoNewWindow -Wait @parameters
      Write-Host "."

}





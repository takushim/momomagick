# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [String]$file = "")

if ($help -eq $true) {
      Write-Host ("Example: {0} -file IMAGE_FILE" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
. $HOME\.venv\gpu\Scripts\Activate.ps1
$script = [IO.Path]::Combine("$HOME\bin\pytrace", "mmtrack.py")
if ($file.length -eq 0){
      $file = (get-item "*_crop_1_deconv.tif")[0]
}
$stem = "{0}" -f (get-item $file).basename
$record = "{0}_track.json" -f $stem

# process!
if (Test-Path $record) {
      $parameters = @{
            FilePath = (Get-command "py.exe")
            ArgumentList = @($script, "-f", $record, $file)
      }
}
else {
      $parameters = @{
            FilePath = (Get-command "py.exe")
            ArgumentList = @($script, $file)
      }
}

Start-Process -NoNewWindow -Wait @parameters


# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [String]$file = "")

if (($help -eq $true) -or ($file.length -eq 0)) {
      Write-Host ("Example: {0} -file IMAGE_FILE" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
$logfile = 'fetch.log'
. $HOME\.venv\gpu\Scripts\Activate.ps1

# select a file if $file is a folder name
if ((get-item $file) -is [System.IO.DirectoryInfo]) {
      $glob = [IO.Path]::Combine($file, "*_crop_1_deconv.tif")
      $file = (get-item $glob)
}

$image = $file
$folder = (get-item $image).DirectoryName
$date = ((get-item $folder).Parent.Parent.Name -split "_")[0]
$record = [IO.Path]::Combine($folder, ("{0}_track.json" -f (get-item $image).basename))

# start logging
Write-Output ("* {0}: {1}" -f (get-date), $folder) | Tee-object -Append -FilePath $logfile

# copy image
$output_image = "{0}_{1}" -f $date, (get-item $image).Name
Write-Output ("-- {0}" -f $image) | Tee-object -Append -FilePath $logfile
Copy-Item $image $output_image

# copy record
if (Test-Path $record) {
      $output_record = "{0}_{1}" -f $date, (get-item $record).Name
      Write-Output ("-- {0}" -f $record) | Tee-object -Append -FilePath $logfile
      Copy-Item $record $output_record
}

Write-Output (".") | Tee-object -Append -FilePath $logfile


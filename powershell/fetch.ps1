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
      $glob = [IO.Path]::Combine($file, "*_crop_1_deconv_res.tif")
      if (Test-Path $glob) {
            $file = (get-item $glob)
      }
      else {
            $glob = [IO.Path]::Combine($file, "*_crop_1_deconv.tif")
            if (Test-Path $glob) {
                  $file = (get-item $glob)
            }
            else{
                  $glob = [IO.Path]::Combine($file, "*_crop_1.tif")
                  $file = (get-item $glob)
            }
      }
}

$image = $file
$folder = (get-item $image).DirectoryName
$date = ((get-item $folder).Parent.Parent.Name -split "_")[0]
$suffix = ((get-item $folder).Parent.Name -split "_")[-1]
$record = [IO.Path]::Combine($folder, ("{0}_track.json" -f (get-item $image).basename))

if (((get-item $folder).Name -split "_")[-1] -match "[0-9]") {
      $counter = "_{0}" -f ((get-item $folder).Name -split "_")[-1]
}

# start logging
Write-Output ("* {0}: {1}" -f (get-date), $folder) | Tee-object -Append -FilePath $logfile

# copy image
$output_image = "{0}_{1}{2}_{3}" -f $date, $suffix, $counter, (get-item $image).Name
Write-Output ("-- {0}" -f $image) | Tee-object -Append -FilePath $logfile
Copy-Item $image $output_image

# copy record
if (Test-Path $record) {
      $output_record = "{0}_{1}" -f $date, (get-item $record).Name
      Write-Output ("-- {0}" -f $record) | Tee-object -Append -FilePath $logfile
      Copy-Item $record $output_record
}

Write-Output (".") | Tee-object -Append -FilePath $logfile


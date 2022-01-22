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
$glob = @("*_crop_1_deconv_res.tif", "*_crop_1_deconv_iso.tif", "*_crop_1_deconv.tif", "*_crop_1.tif")


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
$cellid = ((get-item $folder).Parent.Name -split "_")[-1]
$image_name = (get-item $image).BaseName
$image_ext = (get-item $image).Extension
#$record = [IO.Path]::Combine($folder, ("{0}_track.json" -f (get-item $image).basename))

if (((get-item $folder).Name -split "_")[-1] -match "^[0-9]+$") {
      $counter = ((get-item $folder).Name -split "_")[-1]
}
else{
      $counter = 0
}

# start logging
Write-Output ("* {0}: {1}" -f (get-date), $folder) | Tee-object -Append -FilePath $logfile

# copy image
$output_image = "{0}_{1}_{2}_{3}{4}" -f $date, $cellid, $image_name, $counter, $image_ext
Write-Output ("-- {0}" -f $image) | Tee-object -Append -FilePath $logfile
Copy-Item $image $output_image

# end logging
Write-Output (".") | Tee-object -Append -FilePath $logfile


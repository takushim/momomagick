# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [Switch]$original = $false,
      [String[]]$folders)

if (($help -eq $true) -or ($folders.Count -eq 0)) {
      Write-Host ("Example: {0} -folder FOLDER_LIST" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
$logfile = 'fetch.log'
. $HOME\.venv\gpu\Scripts\Activate.ps1

# select a file if $file is a folder name
if ($original -eq $true) {
      $globs = @(
            "*poc_crop_1.tif",
            "*drift_crop_1.tif",
            "*_crop_1.tif"
            )
}
else {
      $globs = @(
            "*poc_crop_1_deconv_res.tif",
            "*poc_crop_1_deconv_iso.tif",
            "*poc_crop_1_deconv.tif",
            "*poc_crop_1.tif",
            "*drift_crop_1_deconv_res.tif",
            "*drift_crop_1_deconv_iso.tif",
            "*drift_crop_1_deconv.tif",
            "*drift_crop_1.tif",
            "*_crop_1_deconv_res.tif",
            "*_crop_1_deconv_iso.tif",
            "*_crop_1_deconv.tif",
            "*_crop_1.tif"
            )
}

# expand folders
$folders = (get-item $folders)

foreach ($folder in $folders) {
      if ((get-item $folder) -isnot [System.IO.DirectoryInfo]) {
            Write-Host ("{0} is not a folder. Skipping." -f $folder)
            continue
      }

      foreach ($glob in $globs) {
            $glob_path = [IO.Path]::Combine($folder, $glob)
            if (Test-Path $glob_path) {
                  $file = (get-item $glob_path)
                  break
            }
      }
      if ($file.Count -ne 1) {
            Write-Host ("No or multiple images found in {0}. Skipping." -f $folder)
            continue
      }

      $image = $file
      $folder = (get-item $image).DirectoryName
      $date = ((get-item $folder).Parent.Parent.Name -split "_")[0]
      $cellid = ((get-item $folder).Parent.Name -split "_")[-1]
      $image_name = (get-item $image).BaseName
      $image_ext = (get-item $image).Extension
      
      if (((get-item $folder).Name -split "_")[-1] -match "^[0-9]+\w?$") {
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
}





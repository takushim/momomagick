# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [String]$folder = 'analysis',
      [String]$analysis = 'all',
      [Float]$timescale = 1.0,
      [Int]$fitstart = 0,
      [Int]$fitend = 0,
      [String]$meaneach = '',
      [String]$bleach_frame = '0.001',
      [Switch]$runeach = $false,
      [String[]]$files = @())

if ($help -eq $true) {
      Write-Host ("Example: {0} -timescale 10.0 -fitstart 0 -fitend 30 -folder temp -runeach -meaneach 100 -bleach_frame (#|fix|vol) -files A, B, C" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
. $HOME\.venv\gpu\Scripts\Activate.ps1
$scriptpath = "$HOME\bin\dispim"
$script = [IO.Path]::Combine($scriptpath, "mmanalyze_lifetime.py")
if ($files.count -eq 0){
      $files = (get-item "*track.json")
}
$stem = "{0}" -f (get-item $files[0]).basename

if ($analysis -like 'all') {
      $analyses = @("cumulative", "lifetime", "regression", "scatter")
}
else{
      $analyses = @($analysis)
}

if ($bleach_frame -like 'vol'){
      $bleach_frame = 9.75
}
elseif ($bleach_frame -like 'fix') {
      $bleach_frame = 26.57
}
else {
      $bleach_frame = [Float]($bleach_frame)
}

# processing function
function Analyze {
      Param([String[]]$input_files = @())
      foreach ($analysis in $analyses){
            Write-Host ("***** {0} *****" -f $analysis)
      
            if ($input_files.Count -gt 1) {
                  $text = [IO.Path]::Combine($folder, ("Summary_{0}.txt" -f $analysis))
                  $graph = [IO.Path]::Combine($folder, ("Summary_{0}.png" -f $analysis))
            }
            else {
                  $text = [IO.Path]::Combine($folder, ("{0}_{1}.txt" -f $stem, $analysis))
                  $graph = [IO.Path]::Combine($folder, ("{0}_{1}.png" -f $stem, $analysis))
            }

            $arglist = @($script, "-x", $timescale, "-o", $text, "-g", $graph, "-a", $analysis, "-b", $bleach_frame)
            if (($analysis -like 'cumulative') -or ($analysis -like 'lifetime')) {
                  $arglist = $arglist + @("-s", $fitstart, "-e", $fitend)
            }
            elseif ($analysis -like 'scatter') {
                  if ($meaneach -notlike '') {
                        $arglist = $arglist + @("-m", $meaneach)
                  }
            }

            $quoted_files = @()
            foreach ($input_file in $input_files) {
                  $quoted_files = $quoted_files + ("`"{0}`"" -f $input_file)
            }
      
            $parameters = @{
                  FilePath = (Get-command "py.exe")
                  ArgumentList = $arglist + $quoted_files
            }
      
            Start-Process -NoNewWindow -Wait @parameters
            Write-Host "."
      }
      

}

# mkdir and loop!
mkdir -Force $folder | Out-Null
if ($runeach) {
      foreach ($file in $files) {
            Analyze $file
      }
}
else {
      Analyze $files
}
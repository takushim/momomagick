# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [String]$folder = 'analysis',
      [Float]$timescale = 1.0,
      [Int]$fitstart = 0,
      [Int]$fitend = 0,
      [Float]$bleach_frame = 11.95,
      [Switch]$separate = $false,
      [String[]]$files = @())

if ($help -eq $true) {
      Write-Host ("Example: {0} -timescale 10.0 -fitstart 0 -fitend 30 -folder temp -separate -files A, B, C" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
$ErrorActionPreference = 'Stop'
. $HOME\.venv\gpu\Scripts\Activate.ps1
$scriptpath = "$HOME\bin\dispim"
$script = [IO.Path]::Combine($scriptpath, "mmanalyze_lifetime.py")
$analyses = @("cumulative", "lifetime", "regression", "scatter")
if ($files.count -eq 0){
      $files = (get-item "*track.json")
}
$stem = "{0}" -f (get-item $files[0]).basename

# mkdir and loop!
mkdir -Force $folder | Out-Null
foreach ($analysis in $analyses){
      Write-Host ("***** {0} *****" -f $analysis)

      $text = [IO.Path]::Combine($folder, ("{0}_{1}.txt" -f $stem, $analysis))
      $graph = [IO.Path]::Combine($folder, ("{0}_{1}.png" -f $stem, $analysis))
      $arglist = @($script, "-x", $timescale, "-o", $text, "-g", $graph, "-a", $analysis,
                   "-s", $fitstart, "-e", $fitend, "-b", $bleach_frame)

      if ($separate -eq $true) {
            $arglist = $arglist + @("-p")
      }

      $parameters = @{
            FilePath = (Get-command "py.exe")
            ArgumentList = $arglist + $files
      }

      Start-Process -NoNewWindow -Wait @parameters

      Write-Host "."
}
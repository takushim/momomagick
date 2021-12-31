
# get/set parameters from command line arguments
Param([Switch]$help = $false,
      [String]$folder = 'analysis',
      [Float]$timescale = 1.0,
      [String[]]$files = @())

if ($help -eq $true) {
      Write-Host ("Example: {0} -timescale 10.0 -folder temp -files A, B, C" -f $MyInvocation.MyCommand.Name)
      return 0
}

# default values
. $HOME\.venv\gpu\Scripts\Activate.ps1
$scriptpath = "$HOME\bin\dispim"
$script = [IO.Path]::Combine($scriptpath, "mmanalyze_lifetime.py")
$analyses = @("lifetime", "regression", "cumulative", "counting")
if ($files.count -eq 0){
      $files = (get-item "*track.json")
}
$stem = "{0}" -f (get-item $files[0]).basename

# mkdir and loop!
mkdir -Force $folder | Out-Null
foreach ($analysis in $analyses){
      $text = [IO.Path]::Combine($folder, ("{0}_{1}.txt" -f $stem, $analysis))
      $graph = [IO.Path]::Combine($folder, ("{0}_{1}.png" -f $stem, $analysis))
      $parameters = @{
            FilePath = (Get-command "py.exe")
            ArgumentList = @($script, "-x", $timescale, "-o", $text, "-g", $graph, "-a", $analysis) + $files
      }
      Start-Process -NoNewWindow -Wait @parameters
}
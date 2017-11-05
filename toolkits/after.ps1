Set-PSDebug -Trace 1

$MessHome = "C:\MESS\Worker"
$ResultsDir = "$MessHome\Results"
$StdoutPath = "$ResultsDir\out.txt"

Start-Transcript -Path $StdoutPath

$OutProcessListPath = "$ResultsDir\ps.log"
$OutProcessListPath

$OutCsvPath = "$ResultsDir\procmon.csv"
$OutCsvPath

$OutLsPath = "$ResultsDir\ls.txt"
$OutLsPath

$ProcmonPath = Get-ChildItem -Recurse $MessHome | ? -Property Name -eq "Procmon.exe" | Select -ExpandProperty FullName
$ProcmonPath

Invoke-Expression "$ProcmonPath /Terminate"

$PmlPath = Get-ChildItem -Recurse $MessHome | ? -Property Name -like "*.pml" | Select -ExpandProperty FullName
$PmlPath

$PmlToCsvCmd = "$ProcmonPath /OpenLog $PmlPath /SaveAs $OutCsvPath"
$PmlToCsvCmd

Invoke-Expression $PmlToCsvCmd
Start-Sleep -Seconds 60

Get-ChildItem -Recurse C:\Mess | Out-File $OutLsPath
Get-Process | Out-File $OutProcessListPath

Stop-Transcript

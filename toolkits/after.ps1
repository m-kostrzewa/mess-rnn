Set-PSDebug -Trace 1

$MessHome = "C:\MESS\Worker"
$ResultsDir = "$MessHome\Results"
$StdoutPath = "$ResultsDir\out.txt"

Start-Transcript -Path $StdoutPath

$OutPsPath = "$ResultsDir\ps.txt"
$OutPsPath

$OutLsPath = "$ResultsDir\ls.txt"
$OutLsPath

$ProcmonPath = Get-ChildItem -Recurse $MessHome | ? -Property Name -eq "Procmon.exe" | Select -ExpandProperty FullName
$ProcmonPath

Get-ChildItem -Recurse C:\Mess | Out-File -Append $OutLsPath
Get-Date | Out-File -Append $OutPsPath
Get-Process | Out-File -Append $OutPsPath

$PmlPaths = Get-ChildItem -Recurse $MessHome | ? -Property Name -like "*.pml" | Select -ExpandProperty FullName
$PmlPaths

$PmlPaths | % {

    $Filename = (($_ -split "\\")[-1] -split ".pml")[0]
    $Filename

    $OutCsvPath = "$ResultsDir\$Filename.csv"
    $OutCsvPath

    $PmlToCsvCmd = "$ProcmonPath /Quiet /Minimized /AcceptEula /Noconnect /OpenLog $_ /SaveAs $OutCsvPath"
    Invoke-Expression $PmlToCsvCmd

    $Start = Get-Date
    $WaitTimeoutMinutes = 5
    while((Get-Process -Name Procmon) -and (($e = Get-Date) - $Start).Minutes -le $WaitTimeoutMinutes) { 
        Start-Sleep -s 1
    }
}

Stop-Transcript

' This script detects whether we are in headless mode and launches lemonade-server
' either in headless mode or with a system tray icon.

Set wshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Declare headless variable
Dim HEADLESS

' Build argument list from optional environment variables
Dim args, host, port, logLevel, llamaBackend, ctxSize
args = " serve"

host = wshShell.ExpandEnvironmentStrings("%LEMONADE_HOST%")
If host <> "%LEMONADE_HOST%" And host <> "" Then
    args = args & " --host """ & host & """"
End If

port = wshShell.ExpandEnvironmentStrings("%LEMONADE_PORT%")
If port <> "%LEMONADE_PORT%" And port <> "" Then
    args = args & " --port " & port
End If

logLevel = wshShell.ExpandEnvironmentStrings("%LEMONADE_LOG_LEVEL%")
If logLevel <> "%LEMONADE_LOG_LEVEL%" And logLevel <> "" Then
    args = args & " --log-level """ & logLevel & """"
End If

llamaBackend = wshShell.ExpandEnvironmentStrings("%LEMONADE_LLAMACPP%")
If llamaBackend <> "%LEMONADE_LLAMACPP%" And llamaBackend <> "" Then
    args = args & " --llamacpp """ & llamaBackend & """"
End If

ctxSize = wshShell.ExpandEnvironmentStrings("%LEMONADE_CTX_SIZE%")
If ctxSize <> "%LEMONADE_CTX_SIZE%" And ctxSize <> "" Then
    args = args & " --ctx-size " & ctxSize
End If

' Simple GUI detection: check if system tray is available
On Error Resume Next
Set shell = CreateObject("Shell.Application")
If Err.Number = 0 Then
    ' Try to access the system tray area
    Set trayWnd = shell.Windows()
    If Err.Number = 0 Then
        ' GUI mode: show tray
        Set trayWnd = Nothing
        Set shell = Nothing
        On Error GoTo 0
        HEADLESS = False
    Else
        ' Headless mode: no GUI
        Set shell = Nothing
        On Error GoTo 0
        HEADLESS = True
    End If
Else
    ' Headless mode: no GUI
    On Error GoTo 0
    HEADLESS = True
End If

If HEADLESS = True Then
    ' Headless mode: open a terminal and run the server without the tray
    wshShell.Run """" & scriptDir & "\lemonade-server.bat""" & args & " --no-tray", 1, True
Else
    ' Check if we're in CI mode via environment variable
    ciMode = wshShell.ExpandEnvironmentStrings("%LEMONADE_CI_MODE%")
    If ciMode <> "%LEMONADE_CI_MODE%" And (LCase(ciMode) = "true" Or LCase(ciMode) = "1") Then
        ' CI mode: run without tray even in GUI environment
        wshShell.Run """" & scriptDir & "\lemonade-server.bat""" & args & " --no-tray", 1, True
    Else
        ' GUI mode: Run the server on a hidden window with the tray
        wshShell.Run """" & scriptDir & "\lemonade-server.bat""" & args, 0, False
    End If
End If

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$outPath = "c:\Users\asus\OneDrive\Desktop\Projects\minor project\pipeline\sample.wav"
$synth.SetOutputToWaveFile($outPath)
$synth.Speak("Hello everyone, I am so happy to be presenting this project today. This is a demonstration of semantic speech communication.")
$synth.Dispose()
Write-Host "Sample audio created at $outPath"

param([switch]$VerticalSweep, [int]$FramesPerPose = 5)
$ErrorActionPreference = 'Stop'
$sdk = Join-Path ${env:ProgramFiles} 'Microsoft SDKs/Kinect/v1.8/Assemblies/Microsoft.Kinect.dll'
Add-Type -Path $sdk
Add-Type -ReferencedAssemblies $sdk -TypeDefinition @"
using System;
using System.IO;
using System.Threading;
using Microsoft.Kinect;
public static class KinectBridge {
    public static void Run(bool sweep, int framesPerPose) {
        KinectSensor sensor = null;
        foreach (KinectSensor candidate in KinectSensor.KinectSensors)
            if (candidate.Status == KinectStatus.Connected) { sensor = candidate; break; }
        if (sensor == null) throw new IOException("Kinect v1 no conectado.");
        const int n = 640 * 480;
        var depth = new DepthImagePixel[n];
        var color = new byte[n * 4];
        var map = new ColorImagePoint[n];
        var mm = new short[n];
        var coordinates = new short[n * 2];
        var packet = new byte[n * 10 + (sweep ? 4 : 0)];
        var output = Console.OpenStandardOutput();
        var df = DepthImageFormat.Resolution640x480Fps30;
        var cf = ColorImageFormat.RgbResolution640x480Fps30;
        sensor.DepthStream.Enable(df);
        sensor.ColorStream.Enable(cf);
        var ready = new AutoResetEvent(false);
        var gate = new object();
        var waitingDepth = new DepthImagePixel[n];
        var waitingColor = new byte[n * 4];
        int depthFrames = 0, colorFrames = 0, produced = 0, consumed = 0;
        EventHandler<AllFramesReadyEventArgs> receive = (sender, args) => {
            using (var d = args.OpenDepthImageFrame())
            using (var c = args.OpenColorImageFrame()) {
                if (d != null) Interlocked.Increment(ref depthFrames);
                if (c != null) Interlocked.Increment(ref colorFrames);
                if (d == null || c == null) return;
                lock (gate) {
                    d.CopyDepthImagePixelDataTo(waitingDepth);
                    c.CopyPixelDataTo(waitingColor);
                    produced++;
                    ready.Set();
                }
            }
        };
        sensor.AllFramesReady += receive;
        int? initialAngle = null;
        try {
            try { sensor.Start(); }
            catch (Exception error) { throw new IOException("KinectSensor.Start: " + error.Message, error); }
            int[] angles = { 0 };
            if (sweep) {
                initialAngle = sensor.ElevationAngle;
                int center = Math.Max(-10, Math.Min(10, initialAngle.Value));
                angles = new[] { center + 14, center, center - 14 };
            }
            int pose = 0, saved = 0;
            bool move = sweep;
            while (pose < angles.Length) {
                if (move) {
                    sensor.ElevationAngle = angles[pose];
                    Thread.Sleep(1500);
                    lock (gate) { consumed = produced; }
                    move = false;
                }
                if (!ready.WaitOne(6500))
                    throw new IOException(String.Format("Sin pares RGB-D desde AllFramesReady. Eventos profundidad={0}, color={1}.", depthFrames, colorFrames));
                lock (gate) {
                    if (consumed == produced) continue;
                    consumed = produced;
                    Array.Copy(waitingDepth, depth, n);
                    Array.Copy(waitingColor, color, n*4);
                }
                sensor.CoordinateMapper.MapDepthFrameToColorFrame(df, depth, cf, map);
                for (int i = 0; i < n; i++) {
                    mm[i] = (short)depth[i].Depth;
                    coordinates[i * 2] = (short)map[i].X;
                    coordinates[i * 2 + 1] = (short)map[i].Y;
                }
                Buffer.BlockCopy(mm, 0, packet, 0, n * 2);
                Buffer.BlockCopy(color, 0, packet, n * 2, n * 4);
                Buffer.BlockCopy(coordinates, 0, packet, n * 6, n * 4);
                if (sweep) Buffer.BlockCopy(BitConverter.GetBytes(angles[pose]), 0, packet, n*10, 4);
                output.Write(packet, 0, packet.Length);
                output.Flush();
                if (sweep && ++saved >= framesPerPose) {
                    saved = 0;
                    pose++;
                    move = pose < angles.Length;
                }
            }
        } finally {
            sensor.AllFramesReady -= receive;
            if (initialAngle.HasValue && sensor.IsRunning) {
                try { sensor.ElevationAngle = initialAngle.Value; }
                catch (Exception error) { Console.Error.WriteLine("Restaurar inclinacion: " + error.Message); }
            }
            if (sensor.IsRunning) sensor.Stop();
            ready.Dispose();
        }
    }
}
"@
[KinectBridge]::Run($VerticalSweep.IsPresent, $FramesPerPose)

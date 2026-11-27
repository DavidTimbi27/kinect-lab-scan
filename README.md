# Gemelo digital RGB-D del laboratorio

Universidad de Cuenca · Ingeniería en Telecomunicaciones. El flujo captura con Kinect v1, registra vistas RGB-D y exporta geometría métrica para preparar un modelo de propagación WiFi.

## Instalación

Usa Python de 64 bits. Las dependencias directas están fijadas en `requirements.txt`. Ejecuta desde `kinect_lab_scan`.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q
```

Si `py -3.12` no reconoce una instalación por usuario, sustituye la primera línea por esta ruta, si existe.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

En Linux, usa `python3 -m venv .venv` y `.venv/bin/python` para los modos sintético y replay. La captura incluida utiliza Windows PowerShell y el SDK de Microsoft. En el equipo de desarrollo, Windows detectó el Kinect por USB y el SDK 1.8 lo enumeró como conectado.

## Conexión y captura real

1. Instala el [SDK Kinect para Windows 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) si tu equipo carece de él. Conecta el adaptador de corriente/USB del Kinect de Xbox 360 a la fuente y a un puerto USB del equipo.
2. Cierra aplicaciones que estén usando la cámara. Inicia la captura desde un escritorio con OpenCV. Pulsa **S** o **espacio** para guardar y **Q** o **Esc** para salir.
3. Monta el sensor en un trípode y espera un par de segundos antes de guardar. El promedio requiere que la cámara y los objetos permanezcan quietos mientras se adquieren los cinco pares.
4. Rota o desplaza el sensor con aproximadamente **60–70 % de solapamiento**. Incluye esquinas, mesas y equipos con estructura en cada vista. Evita vidrio, metal pulido, pantallas apagadas y luz solar directa. Captura explícitamente piso, techo, paredes opuestas y obstáculos. Regresa a una vista inicial para facilitar el cierre de lazo.
5. Mide y anota al menos una referencia con flexómetro, como el ancho de una puerta. Compara la distancia entre esos mismos puntos de la nube con la medida física. Repite la comparación en más de una dirección si vas a inferir dimensiones para simulación.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m src capture laboratorio --average 5
& $py -m src all laboratorio --simulation
& $py -m src visualize laboratorio
```

`all laboratorio --source kinect --simulation` captura y procesa en una sola ejecución. El nombre de una captura nueva debe ser único. Para un diagnóstico sin abrir una ventana, usa `capture prueba_usb --auto --count 1 --average 5`.

Para repetir el barrido vertical motorizado desde una base fija, usa una sesión nueva. El Kinect toma tres posiciones, espera entre movimientos e intenta volver a la inclinación de partida al terminar.

\`\`\`powershell
& $py -m src capture barrido_02 --vertical-sweep --average 5
& $py -m src reconstruct barrido_02 --method rgbd
\`\`\`

## Prueba sin sensor y reproducción

```powershell
$py = ".\.venv\Scripts\python.exe"
$env:OMP_NUM_THREADS = "4"
& $py -m src all ensayo --source synthetic --simulation
& $py -m src all ensayo_odo --source synthetic --method rgbd --simulation
& $py -m src capture copia --source replay --input-session ensayo
& $py -m src reconstruct copia --method fpfh
& $py -m src geometry copia --simulation
```

Para una prueba de principio a fin con visor en el navegador, usa un nombre de sesión nuevo.

```powershell
& $py -m src all demostracion --source synthetic --method rgbd --simulation
& $py -m src.web_viewer demostracion --voxel-mm 10
Invoke-Item .\output\demostracion\visor_3d.html
```

La escena sintética mide **4,40 × 4,00 × 2,60 m**. Incluye cajas, textura y una trayectoria con retorno. Su pose de referencia queda en el manifiesto para validación. El registro se estima a partir de las imágenes y las nubes.

## Calibración y registro

- Profundidad de 16 bits en milímetros axiales y color PNG sobre la cuadrícula de profundidad. Cada entrada en `data/raw/<sesión>/metadata.json` identifica los archivos, el índice, la hora UTC, la alineación, los intrínsecos y la cantidad de frames promediados.
- El SDK convierte el flujo nativo a milímetros y usa el [CoordinateMapper](https://learn.microsoft.com/en-us/previous-versions/windows/kinect-1.8/jj663779%28v%3Dieb.10%29) con la calibración de fábrica RGB/profundidad. Para proyectar la nube se usa el modelo pinhole nominal del SDK a 640×480, `fx = fy = 571.26`, `cx = 320`, `cy = 240`. El focal proviene de `NuiImageCamera.h` a 320×240 y se multiplica por dos. Mide la escala y sustituye los intrínsecos por los de tu unidad si el error excede tu tolerancia.
- La profundidad se recorta de **500 a 4000 mm**. Cero representa falta de medición. El valor 2047 es un centinela en el flujo de disparidad cruda de ciertos backends. En el flujo métrico del SDK, 2047 mm es una distancia válida. Si tu fuente métrica importada reserva ese número, configura `depth.invalid_mm: [0, 2047]`. Convierte la disparidad a milímetros antes de construir un manifiesto replay.
- Para usar una calibración RGB-D propia, copia `config.yaml`, actualiza `capture.depth_intrinsic` y añade `capture.color_registration` con `rgb_intrinsic` (width, height, fx, fy, cx, cy), `rotation_depth_to_rgb` (matriz 3×3), `translation_depth_to_rgb_m` (vector de 3 componentes) y, opcionalmente, `rgb_distortion` en el orden OpenCV. Usa intrínsecos compatibles con la cuadrícula de profundidad y el flujo RGB de 640×480. Pasa `--config calibracion.yaml` antes del subcomando.
- El método principal usa FPFH + RANSAC para inicializar el registro sin una pose previa precisa y después refina por ICP punto a plano y con color. En tomas consecutivas con movimiento pequeño y textura distinguible, `--method rgbd` usa odometría RGB-D como inicialización. Ambos métodos registran fitness, RMSE, matriz origen → destino y aceptación. El grafo evalúa cierres de lazo y optimiza las poses globales. [Registro multivista](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/multiway_registration.html) · [Odometría RGB-D](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html)
- La fusión conserva metros. La geometría usa el piso como Z=0, paredes dominantes para orientar los ejes y planos opuestos para medir dimensiones. El informe identifica cuándo una dimensión proviene de la extensión observada. Para mejorar la estimación, pon el Kinect aproximadamente nivelado en la primera vista o ajusta `geometry.up_hint_camera` al vector hacia arriba en esa cámara.

## Archivos generados

En `output/<sesión>/` quedan:

| Archivo | Contenido |
| --- | --- |
| `nube_global.ply`, `nube_global.pcd` | Nube RGB fusionada en el marco de la primera captura |
| `nube_piso.ply`, `nube_piso.pcd` | Nube alineada al recinto, con Z hacia arriba |
| `malla.obj`, `malla.ply` | Malla simplificada por ball pivoting, con posibles aberturas en zonas sin datos |
| `geometria.json` | Piso, techo, paredes, obstáculos planos, dimensiones, cajas envolventes y transformación |
| `modelo_simulacion.json` | Con `--simulation`, parches planos con soporte y límites en metros para importar a un simulador |
| `pose_graph.json`, `registro_pares.json` | Grafo optimizado y diagnóstico por par |
| `planta.png`, `frontal.png`, `lateral.png`, `isometrica.png` | Vistas ortográficas en PNG, incluso sin escritorio |
| `reporte.json`, `reporte.md` | Parámetros, tiempos, métricas y dimensiones |

## Visor 3D en el navegador

Genera un HTML autónomo a partir de una sesión reconstruida. El visor permite rotar, acercar, cambiar de vista, mostrar los ejes en metros y ajustar el tamaño de los puntos.

```powershell
& $py -m src.web_viewer barrido_vertical
```

Abre output/barrido_vertical/visor_3d.html en un navegador compatible con WebGL.

Para un visor más denso, agrega `--voxel-mm 10` al comando. Empieza con puntos de 1,8 px y agrándalos si la nube se ve dispersa. La nitidez de los bordes también depende de la alineación y del solapamiento entre vistas.

## Solución de problemas

- **El SDK no recibe pares.** Comprueba alimentación y USB, prueba otro puerto y cierra el visor del SDK u otra aplicación de cámara. Desconecta y vuelve a conectar el sensor si el SDK sigue sin entregar datos. `usbipd list` puede confirmar la presencia del dispositivo. Si usas WSL2 para procesar, captura con Windows y comparte la carpeta de datos. La captura directa con libfreenect en WSL2 requeriría transferir el USB con `usbipd` y usar otro backend.
- **Fitness bajo o registro incorrecto.** Abre `registro_pares.json` para localizar la transición. Repite esas vistas con pasos más cortos y objetos reconocibles. En un recinto con paredes muy parecidas, un fitness alto puede coexistir con una pose incorrecta. Revisa el cierre y la referencia métrica.
- **Faltan planos o la malla tiene agujeros.** Captura el piso o techo con inclinación y solapamiento suficiente. Ajusta `geometry.plane_threshold_m` o `geometry.mesh_voxel_m` según el ruido y densidad. Comprueba la nube alineada y el informe antes de usar el modelo.
- **No abre el visor.** Ejecuta `visualize` desde una sesión con escritorio y controladores gráficos. Las cuatro imágenes PNG se generan en CPU aunque no se abra la ventana.

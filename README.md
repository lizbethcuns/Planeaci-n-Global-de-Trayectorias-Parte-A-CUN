# Planeaci-n-Global-de-Trayectorias-Parte-A-CUN
### Planeación Global de Trayectorias - Parte A
En esta tarea se realizo una Planeacion Global de trayectoriass, en este caso se uso el parametro LPA* y rrt para el mapa de Oschersleben con una separacion de 0.5m y 1m cada uno.
- Descripcion del parametro LPA*

El algoritmo Lifelong Planning A* (LPA*) es un método de planificación de rutas incremental utilizado en robótica móvil para encontrar el camino óptimo entre un punto inicial y uno final dentro de un entorno discretizado. A diferencia de A* tradicional, LPA* reutiliza información de búsquedas anteriores, lo que le permite recalcular rutas de manera más eficiente cuando cambian las condiciones del mapa o cuando se actualizan los costos de los nodos. Esto lo hace especialmente adecuado para entornos parcialmente dinámicos, donde no es necesario replantear toda la ruta desde cero, sino solo actualizar las zonas afectadas. En la implementación utilizada, el algoritmo trabaja sobre una representación tipo Grid, mantiene estructuras de costos consistentes y permite obtener trayectorias óptimas que luego pueden ser exportadas para su uso en sistemas de navegación o control.
####Explicaicon de la ejecucion del LPA para 0.5m y 1m
Una vez obtenida la ruta óptima mediante LPA*, se implementó un proceso adicional de remuestreo de la trayectoria con el objetivo de generar waypoints separados uniformemente cada 0.5 metros. Este paso es fundamental para aplicaciones de seguimiento de trayectoria, ya que proporciona puntos equidistantes que facilitan el control del vehículo.

El remuestreo se realizó en coordenadas del mundo real, calculando la distancia acumulada entre puntos consecutivos y generando nuevos puntos intermedios cuando se alcanza la separación deseada.

####1. Carga y preparación del mapa
¿Qué se hizo?
Se carga el mapa desde un archivo YAML, se lee la imagen asociada y se prepara para ser utilizada por el algoritmo LPA*. Esto incluye la binarización del mapa, el engrosamiento de obstáculos y la reducción de resolución (downsampling) para mejorar el rendimiento computacional.
```html
def load_map(yaml_path, downsample_factor=1):
```
Define la función encargada de cargar y procesar el mapa.
```html
with yaml_path.open('r') as f:
    map_config = yaml.safe_load(f)
```
Lee el archivo YAML que contiene la información del mapa (imagen, resolución y origen).
```html
map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
```
Carga la imagen del mapa en escala de grises.
```html
map_bin = np.zeros_like(map_img, dtype=np.uint8)
map_bin[map_img < int(0.45 * 255)] = 1
```
Binariza el mapa:
- 1 representa obstáculos
- 0 representa espacio libre
####2. Engrosamiento de obstáculos (Dilatación)
####¿Qué se hizo?
Se aplica una dilatación morfológica para engrosar los obstáculos. Esto evita que el planificador genere rutas demasiado cercanas a los bordes de la pista.
```html
if downsample_factor > 12:
    map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
elif downsample_factor >= 4:
    map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
```
Dependiendo del nivel de reducción del mapa, se aplica una dilatación con un kernel de 5x5 para simular el ancho real del vehículo y aumentar la seguridad de la trayectoria.
####3. Downsampling del mapa y ajuste de resolución
####¿Qué se hizo?
Se reduce el tamaño del mapa para disminuir la cantidad de nodos que debe procesar LPA*. Luego, se ajusta la resolución para mantener la coherencia con el mundo real.
```html
map_bin = map_bin.astype(np.float32)
h, w = map_bin.shape
map_bin = cv2.resize(
    map_bin,
    (w // downsample_factor, h // downsample_factor),
    interpolation=cv2.INTER_AREA
)
```
Reduce el tamaño del mapa usando interpolación por área.
```html
if downsample_factor > 12:
    map_bin = (map_bin > 0.10).astype(np.uint8)
elif downsample_factor >= 4:
    map_bin = (map_bin > 0.25).astype(np.uint8)
else:
    map_bin = (map_bin >= 0.5).astype(np.uint8)
```
Re-binariza el mapa después del downsampling.
```html
resolution *= downsample_factor
```
Ajusta la resolución para que cada celda del mapa represente correctamente la distancia real.
####4. Creación del entorno Grid para LPA*
####¿Qué se hizo?
Se transforma el mapa binario en un entorno tipo Grid, que es el formato requerido por el algoritmo LPA* dentro de python_motion_planning.
```html
def grid_from_map(map_bin):
    h, w = map_bin.shape
    env = Grid(w, h)
```
Inicializa un entorno de cuadrícula con el tamaño del mapa.
```html
obstacles = {
    (x, h - 1 - y)
    for y in range(h)
    for x in range(w)
    if map_bin[y, x] == 1
}
```
Convierte cada pixel ocupado en un obstáculo dentro del Grid.
```html
env.update(obstacles)
```
Actualiza el entorno con todos los obstáculos detectados.
####5. Conversión de coordenadas
Se implementaron funciones para convertir coordenadas del mundo real a coordenadas del mapa y viceversa, necesarias para definir start, goal y exportar la ruta.
```html
def world_to_map(x_world, y_world, resolution, origin):
```
Convierte coordenadas reales a índices del mapa.
```html
return (
    int((x_world - origin[0]) / resolution),
    int((y_world - origin[1]) / resolution)
)
```
####6. Ejecución del algoritmo LPA*
```html
planner = SearchFactory()("lpa_star", start=start, goal=goal, env=env)
```
Inicializa el planificador LPA* usando el entorno Grid.
```html
planner.run()
cost, path_map, _ = planner.plan()
```
####7. Remuestreo de la ruta a 0.5 metros
Se implementó una función adicional para remuestrear la ruta generada por LPA*, de modo que los waypoints estén separados cada 0.5 metros en el mundo real.
```html
step = 0.5  # 0.5 METROS
```
Define la distancia deseada entre waypoints.
```html
seg_len = math.hypot(dx, dy)
```
Calcula la longitud de cada segmento de la ruta.
```html
while acc_dist + seg_len >= step:
```
Inserta nuevos puntos intermedios cuando se alcanza la distancia deseada.
```html
resampled.append((xn, yn))
```
####8. Guardado de la trayectoria final
La trayectoria remuestreada se guarda en un archivo CSV para su uso posterior en simulación o control.
```html
save_path_world_csv(path_05m, "lpastar_05m.csv")
```
Se obtuvo una trayectoria óptima calculada mediante LPA*, procesada para respetar la geometría del mapa y finalmente remuestreada a intervalos constantes de 0.5 m, lista para ser utilizada en sistemas de navegación y control.

#### Ahora para RRT 

#### 1. Carga y binarización del mapa
¿Qué se hizo?

Se carga el mapa desde un archivo YAML, se lee la imagen asociada y se convierte en un mapa binario donde los obstáculos y el espacio libre quedan claramente definidos.
```html
def load_map(yaml_path, downsample_factor=1):
with yaml_path.open('r') as f:
    map_config = yaml.safe_load(f)
map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
map_bin = np.zeros_like(map_img, dtype=np.uint8)
map_bin[map_img < int(0.45 * 255)] = 1

```
#### 2. Reducción de resolución (Downsampling)
¿Qué se hizo?

Se reduce la resolución del mapa para disminuir el costo computacional del algoritmo RRT.
```html
map_bin = cv2.resize(
    map_bin,
    (w // downsample_factor, h // downsample_factor),
    interpolation=cv2.INTER_AREA
)
resolution *= downsample_factor
```
#### 3. Conversión del mapa a entorno RRT
Se transforma el mapa binario en un entorno compatible con el planificador RRT, definiendo los obstáculos como celdas ocupadas.
```html
def map_from_binary(map_bin):
env = Map(w, h)
env.update(obs_rect=obs_rect)
```
#### 4. Conversión de coordenadas mundo ↔ mapa
Se convierten las coordenadas del mundo real a coordenadas del mapa y viceversa para poder planificar correctamente.
```html
def world_to_map(x_world, y_world, resolution, origin):
def map_to_world(x_map, y_map, resolution, origin):
```
#### 5. Ejecución del algoritmo RRT
¿Qué se hizo?
Se configura y ejecuta el algoritmo RRT para encontrar una ruta entre el punto inicial y el objetivo.
```html
planner = SearchFactory()(
    "rrt",
    start=start,
    goal=goal,
    env=env,
    max_dist=10,
    sample_num=100000
)
```
#### 6. Exportación de la ruta a CSV
```html
save_path_as_csv(path_05, "rrt_05m.csv", resolution, origin)
```

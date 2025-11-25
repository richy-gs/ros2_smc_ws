# MAV Formation Control Package

Este paquete implementa un sistema de control de formaciones para MAVs (Micro Aerial Vehicles) usando ROS2 Humble.

## Arquitectura

El sistema está compuesto por tres nodos principales:

1. **virtual_leader_trajectory_node**: Genera la trayectoria del líder virtual a partir de waypoints definidos en un archivo de texto.

2. **formation_controller_node**: Controla la posición de los drones líderes alrededor del líder virtual, manteniendo una formación específica.

3. **follower_controller_node**: Controla la posición de los drones seguidores alrededor de sus respectivos líderes.

## Visualización en RViz2

- **Esfera verde**: Líder virtual
- **Esferas opacas (colores)**: Drones líderes (rojo, azul, etc.)
- **Esferas transparentes (mismo color que su líder)**: Drones seguidores

## Instalación

1. Compilar el paquete:
```bash
cd ~/robotarium/ros2_smc_ws
colcon build --packages-select mav_formation_control
source install/setup.bash
```

## Uso

### Lanzar el sistema completo:

```bash
ros2 launch mav_formation_control formation_control.launch.py
```

### Parámetros configurables:

- `waypoint_file`: Archivo de waypoints (default: `config/waypoints.txt`)
- `trajectory_speed`: Velocidad del líder virtual (m/s, default: 1.0)
- `num_leaders`: Número de drones líderes (default: 2)
- `followers_per_leader`: Número de seguidores por líder (default: 2)
- `formation_radius`: Radio de formación alrededor del líder virtual (m, default: 2.0)
- `follower_radius`: Radio de seguidores alrededor de su líder (m, default: 1.0)
- `frame_id`: Frame de referencia TF (default: `map`)
- `publish_rate`: Frecuencia de publicación (Hz, default: 50.0)

### Ejemplo con parámetros personalizados:

```bash
ros2 launch mav_formation_control formation_control.launch.py \
  num_leaders:=2 \
  followers_per_leader:=2 \
  trajectory_speed:=1.5 \
  formation_radius:=3.0
```

### Visualizar en RViz2:

```bash
ros2 launch mav_formation_control rviz.launch.py
```

O manualmente:
```bash
rviz2
```

Y suscribirse a los tópicos:
- `/virtual_leader/marker`
- `/formation_leaders/markers`
- `/formation_followers/markers`

## Formato del archivo de waypoints

El archivo `config/waypoints.txt` contiene las coordenadas de los waypoints, uno por línea:
```
x y z
```

Ejemplo:
```
0.0 0.0 1.0
5.0 0.0 1.0
5.0 5.0 1.0
0.0 5.0 1.0
```

Las líneas que comienzan con `#` son comentarios.

## Tópicos publicados

- `/virtual_leader/pose`: Pose del líder virtual
- `/virtual_leader/marker`: Marcador de visualización del líder virtual
- `/leader_<i>/pose`: Pose del líder i
- `/formation_leaders/markers`: Marcadores de visualización de líderes
- `/follower_<leader_id>_<follower_id>/pose`: Pose del seguidor
- `/formation_followers/markers`: Marcadores de visualización de seguidores



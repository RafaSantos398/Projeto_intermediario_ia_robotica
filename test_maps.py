"""
Testes do planejador A* nos 5 mapas.

Uso:
    python test_maps.py        (ou: pytest test_maps.py)

As imagens são salvas em resultados/.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from astar import AStarPathfinder, prep_map, WALL, UNKNOWN

MAPS = [f'map{i}.pgm' for i in range(1, 6)]
START = (60, 20)
GOAL = (60, 120)
PARAMS = dict(wall_influence=10.0, buffer_factor=3.0)
OUT_DIR = 'resultados'


def plan(map_file, start):
    """Roda as mesmas etapas do run(), mas guarda os resultados intermediários para os asserts."""
    astar = AStarPathfinder(prep_map(map_file), start, GOAL, **PARAMS)
    came_from, final_node = astar.find_path()
    assert final_node is not None, f'{map_file}: nenhum caminho encontrado'
    full_path = astar.reconstruct_path(came_from, final_node)
    known_path = astar.know_path(full_path)
    simplified = astar.simplify_path(known_path)
    return astar, full_path, known_path, simplified


def segment_cells(a, b):
    """Pixels cobertos pela reta de a até b (mesma amostragem do segment_is_safe)."""
    n = int(np.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2)) + 1
    rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return list(zip(rows, cols))


def check(map_file, astar, full_path, known_path, simplified, start):
    m, r = astar.map_array, astar.robot_radius_px

    assert full_path[0] == start, f'{map_file}: caminho completo não começa no início'
    assert simplified[0] == start, f'{map_file}: caminho simplificado não começa no início'
    assert simplified[-1] == known_path[-1], f'{map_file}: simplificação mudou o ponto final'

    for p in full_path:
        assert m[p] != WALL, f'{map_file}: caminho completo passa na parede em {p}'
        assert astar.dist_to_wall[p] >= r, f'{map_file}: caminho completo a {astar.dist_to_wall[p]:.1f}px da parede em {p}'

    for p in known_path:
        assert m[p] != UNKNOWN, f'{map_file}: caminho devolvido entra no desconhecido em {p}'

    for a, b in zip(simplified, simplified[1:]):
        for p in segment_cells(a, b):
            assert m[p] != WALL, f'{map_file}: segmento {a}->{b} cruza parede em {p}'
            assert astar.dist_to_wall[p] >= r, f'{map_file}: segmento {a}->{b} a {astar.dist_to_wall[p]:.1f}px da parede em {p}'
            assert m[p] != UNKNOWN, f'{map_file}: segmento {a}->{b} passa no desconhecido em {p}'


def save_plot(filename, title, astar, full_path, simplified, start):
    rows = max(p[0] for p in full_path + [GOAL]) + 15
    cols = max(p[1] for p in full_path + [GOAL]) + 15

    plt.figure(figsize=(12, 7))
    plt.imshow(astar.map_array[:rows, :cols], cmap='gray', vmin=0, vmax=255)
    unsafe = np.ma.masked_where(~(astar.unsafe & (astar.map_array != WALL)), np.ones_like(astar.map_array))
    plt.imshow(unsafe[:rows, :cols], cmap='autumn', alpha=0.5, vmin=0, vmax=1)

    fy, fx = zip(*full_path)
    plt.plot(fx, fy, color='magenta', linewidth=1, label='Caminho completo (A*)')
    sy, sx = zip(*simplified)
    plt.plot(sx, sy, color='red', linewidth=2, linestyle='--', marker='o', markersize=4, label='Caminho simplificado')
    plt.scatter(start[1], start[0], color='green', s=100, zorder=5, label='Início')
    plt.scatter(GOAL[1], GOAL[0], color='blue', s=100, zorder=5, label='Objetivo')
    plt.plot([], [], color='orange', linewidth=6, alpha=0.5, label=f'Zona de segurança (< {astar.robot_radius_px}px)')

    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close()


def test_each_map_from_fixed_start():
    for map_file in MAPS:
        astar, full_path, known_path, simplified = plan(map_file, START)
        check(map_file, astar, full_path, known_path, simplified, START)
        save_plot(map_file.replace('.pgm', '.png'),
                  f'{map_file}: {len(simplified)} waypoints, objetivo alcançável={astar.GOAL_REACHEABLE}',
                  astar, full_path, simplified, START)
        print(f'{map_file}: OK, {len(simplified)} waypoints -> {simplified}')


def test_exploration_sequence():
    """O robô para no fim do caminho de um mapa e replaneja dali no mapa seguinte."""
    start = START
    for map_file in MAPS:
        astar, full_path, known_path, simplified = plan(map_file, start)
        check(map_file, astar, full_path, known_path, simplified, start)
        save_plot('exploracao_' + map_file.replace('.pgm', '.png'),
                  f'Exploração {map_file}: início {start} -> parada {simplified[-1]}',
                  astar, full_path, simplified, start)
        print(f'Exploração {map_file}: {start} -> {simplified[-1]}')
        start = simplified[-1]


def test_start_inside_safety_zone():
    """Robô parado colado na parede: o planejador deve deixá-lo sair em vez de falhar."""
    start = (64, 20)  # 2px da parede de baixo do map5
    astar = AStarPathfinder(prep_map('map5.pgm'), start, GOAL, **PARAMS)
    assert astar.unsafe[start]
    came_from, final_node = astar.find_path()
    assert final_node is not None, 'não saiu da zona de segurança'
    path = astar.reconstruct_path(came_from, final_node)
    assert path[0] == start
    # Enquanto está na zona de segurança, cada passo só pode afastar (ou manter) a distância da parede.
    for a, b in zip(path, path[1:]):
        if not astar.unsafe[a]:
            break
        assert astar.dist_to_wall[b] >= astar.dist_to_wall[a]
    print(f'Saída da zona de segurança: OK a partir de {start}')


if __name__ == '__main__':
    test_each_map_from_fixed_start()
    test_exploration_sequence()
    test_start_inside_safety_zone()
    print(f'Todos os testes passaram. Imagens em {OUT_DIR}/')

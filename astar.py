import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from PIL import Image
import math

# Valores das células depois do prep_map()
WALL = 0
UNKNOWN = 128
FREE = 255

# Cores usadas no GIF (RGB)
CLOSED_COLOR = (120, 170, 255)   # nós já expandidos
OPEN_COLOR = (255, 210, 80)      # fronteira (fila aberta)
CURRENT_COLOR = (255, 0, 0)      # nó sendo expandido agora

# 8 vizinhos: (dlinha, dcoluna, custo do movimento)
NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

class AStarPathfinder:
    def __init__(self, map_array: np.array, start: tuple, goal: tuple, wall_influence=5.0, buffer_factor=2.0,
                 robot_radius_px=3, unknown_cost=1.0, frontier_margin_px=3):
        """
        Inicializa o A* com mapa, ponto inicial, objetivo e parâmetros de influência.

        Args:
            map_array (np.array): Mapa binário (obstáculos e caminho livre).
            start (tuple): Ponto inicial (linha, coluna).
            goal (tuple): Ponto objetivo (linha, coluna).
            wall_influence (float): Peso da proximidade das paredes.
            buffer_factor (float): Escala da influência das paredes.
            robot_radius_px (float): Distância mínima (em pixels) entre o centro do robô e qualquer parede.
                Células mais próximas que isso são intransponíveis. Depende da resolução do mapa
                (metros/pixel) e do tamanho do robô: raio_px = (raio_robo_m + folga_m) / resolucao_m_por_px.
            unknown_cost (float): Custo extra por passo em célula desconhecida (faz o robô preferir o conhecido).
            frontier_margin_px (int): Quantos pontos o caminho recua antes da fronteira com o desconhecido.
        """
        self.start = tuple(int(v) for v in start)
        self.goal = tuple(int(v) for v in goal)
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.robot_radius_px = robot_radius_px
        self.unknown_cost = unknown_cost
        self.frontier_margin_px = frontier_margin_px
        self.GOAL_REACHEABLE = False

        # Estado da gravação do GIF (preenchido em find_path(record=True)).
        self.frames = []
        self._canvas = None

        # Prepara o mapa, expandindo suas bordas e ajustando o array.
        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)

        # Cria um campo potencial baseado no mapa para influenciar o caminho.
        self.potential_field = self.create_potential_field()


    def preprocess_map(self, map_array: np.array) -> np.array:
        """
        Ajusta o mapa, convertendo valores intermediários para obstáculos.

        Args:
            map_array (np.array): Mapa original.

        Returns:
            np.array: Mapa processado.
        """
        processed = map_array.copy().astype(np.uint8)
        # Qualquer valor que não seja desconhecido nem livre vira parede (postura conservadora).
        processed[(processed != UNKNOWN) & (processed != FREE)] = WALL
        return processed

    def create_potential_field(self) -> np.array:
        """
        Gera campo potencial com base na distância de obstáculos.

        Returns:
            np.array: Campo potencial.
        """
        # Distância euclidiana de cada célula até a parede mais próxima.
        # Só parede conta como obstáculo: o desconhecido não repele o robô.
        self.dist_to_wall = distance_transform_edt(self.map_array != WALL)

        # Zona de segurança: perto demais da parede para o centro do robô.
        self.unsafe = self.dist_to_wall < self.robot_radius_px

        # Custo que decai exponencialmente com a distância além do raio mínimo:
        # vale wall_influence na borda da zona de segurança e cai pela metade
        # a cada ~0.7 * buffer_factor pixels.
        clearance = np.maximum(self.dist_to_wall - self.robot_radius_px, 0.0)
        field = self.wall_influence * np.exp(-clearance / self.buffer_factor)
        field[self.unsafe] = np.inf
        return field

    def heuristic(self, a: tuple, b: tuple) -> float:
        """
        Calcula a heurística entre dois pontos.

        Args:
            a (tuple): Ponto A.
            b (tuple): Ponto B.

        Returns:
            float: Resultado da heurística.
        """
        # Euclidiana: nunca superestima o custo real (passo >= comprimento do movimento).
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def in_bounds(self, node: tuple) -> bool:
        rows, cols = self.map_array.shape
        return 0 <= node[0] < rows and 0 <= node[1] < cols

    def nearest_safe_cell(self, node: tuple) -> tuple:
        """Célula transitável mais próxima de node (usada se o objetivo cair na zona de segurança)."""
        safe_rows, safe_cols = np.nonzero(~self.unsafe)
        if len(safe_rows) == 0:
            return None
        i = np.argmin((safe_rows - node[0]) ** 2 + (safe_cols - node[1]) ** 2)
        return (int(safe_rows[i]), int(safe_cols[i]))

    # ------------------------------------------------------------------
    # Gravação do GIF
    # ------------------------------------------------------------------
    def _base_canvas(self) -> np.array:
        """Imagem RGB do mapa: parede preta, desconhecido cinza, livre branco."""
        canvas = np.zeros((*self.map_array.shape, 3), np.uint8)
        canvas[self.map_array == UNKNOWN] = (128, 128, 128)
        canvas[self.map_array == FREE] = (255, 255, 255)
        return canvas

    def _snapshot(self, canvas: np.array, current: tuple) -> np.array:
        """Cópia do canvas com início, objetivo e nó atual destacados."""
        img = canvas.copy()
        img[self.start] = (0, 200, 0)
        img[self.goal] = (0, 0, 255)
        img[current] = CURRENT_COLOR
        return img

    def save_gif(self, filename: str, path: list = None, scale=4, fps=25, hold_final=30):
        """
        Salva o GIF do processo de busca (requer find_path(record=True)).

        Args:
            filename (str): Caminho do GIF de saída.
            path (list): Caminho a ser desenhado por cima no frame final.
            scale (int): Quantos pixels do GIF representam 1 pixel do mapa.
            fps (int): Frames por segundo.
            hold_final (int): Quantos frames o resultado final fica parado no fim.
        """
        if not self.frames or self._canvas is None:
            print("Nenhum frame gravado.")
            return

        final = self._canvas.copy()

        # Recorta na região explorada, para não mostrar o padding desconhecido inteiro.
        closed = np.all(final == CLOSED_COLOR, axis=2)
        rows, cols = np.nonzero(closed)
        if len(rows) == 0:
            r0, r1, c0, c1 = 0, final.shape[0], 0, final.shape[1]
        else:
            m = 10  # margem do recorte
            r0, r1 = max(rows.min() - m, 0), min(rows.max() + m, final.shape[0])
            c0, c1 = max(cols.min() - m, 0), min(cols.max() + m, final.shape[1])

        def prep(img):
            img = img[r0:r1, c0:c1]
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        frames = [prep(f) for f in self.frames]

        # Frame final com o caminho desenhado por cima.
        last = prep(self._snapshot(final, self.goal))
        if path:
            pts = np.array([[(c - c0 + 0.5) * scale, (r - r0 + 0.5) * scale] for r, c in path],
                           np.int32).reshape(-1, 1, 2)
            cv2.polylines(last, [pts], False, (255, 0, 255), max(1, scale // 2))
        frames += [last] * hold_final

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(filename, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        print(f"GIF salvo em {filename} ({len(imgs)} frames)")

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------
    def find_path(self, record=False, frame_every=30):
        """
        Executa o algoritmo A* para encontrar caminho até o objetivo.

        Args:
            record (bool): Se True, grava frames da busca para o GIF.
            frame_every (int): Grava um frame a cada N nós expandidos.

        Returns:
            dict: Predecessores dos nós no caminho. Se o caminho não for encontrado, retorna None.
            tuple: O ponto final (objetivo) ou None se não encontrado.
        """
        start, goal = self.start, self.goal
        if not self.in_bounds(start) or not self.in_bounds(goal):
            print("Início ou objetivo fora do mapa")
            return None, None

        # Objetivo colado na parede: mira na célula segura mais próxima.
        if self.unsafe[goal]:
            goal = self.nearest_safe_cell(goal)
            if goal is None:
                print("Caminho não encontrado")
                return None, None
            print(f"Objetivo na zona de segurança, usando {goal}")
            self.goal = goal

        dist = self.dist_to_wall
        g_score = {start: 0.0}
        came_from = {}
        open_heap = [(self.heuristic(start, goal), 0.0, start)]
        closed = set()

        self.frames = []
        n_expanded = 0
        if record:
            self._canvas = self._base_canvas()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                if record:
                    self.frames.append(self._snapshot(self._canvas, current))
                return came_from, current
            closed.add(current)

            if record:
                self._canvas[current] = CLOSED_COLOR
                n_expanded += 1
                if n_expanded % frame_every == 0:
                    self.frames.append(self._snapshot(self._canvas, current))

            escaping = self.unsafe[current]
            for dr, dc, step in NEIGHBORS:
                nb = (current[0] + dr, current[1] + dc)
                if not self.in_bounds(nb) or nb in closed:
                    continue
                if self.map_array[nb] == WALL:
                    continue
                # Diagonal não pode cortar quina de parede.
                if dr != 0 and dc != 0 and (self.map_array[current[0] + dr, current[1]] == WALL
                                            or self.map_array[current[0], current[1] + dc] == WALL):
                    continue

                if self.unsafe[nb]:
                    # Só entra na zona de segurança se já está nela (início colado na parede)
                    # e sem se aproximar mais da parede: é o jeito de sair dela.
                    if not escaping or dist[nb] < dist[current]:
                        continue
                    extra = self.wall_influence * math.exp((self.robot_radius_px - dist[nb]) / self.buffer_factor)
                else:
                    extra = self.potential_field[nb]

                if self.map_array[nb] == UNKNOWN:
                    extra += self.unknown_cost

                new_g = g + step + extra
                if new_g < g_score.get(nb, math.inf):
                    g_score[nb] = new_g
                    came_from[nb] = current
                    heapq.heappush(open_heap, (new_g + self.heuristic(nb, goal), new_g, nb))
                    if record:
                        self._canvas[nb] = OPEN_COLOR

        print("Caminho não encontrado")
        return None, None

    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """
        Reconstrói o caminho a partir do ponto final até o inicial.

        Args:
            came_from (dict): O dicionário de predecessores no caminho.
            current (tuple): O ponto final (objetivo).

        Returns:
            list: Lista de tuplas com caminho reconstruído.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def know_path(self, path: list) -> list:
        """
        Remove trechos desconhecidos e ajusta o caminho, se necessário.

        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho ajustado.
        """
        self.GOAL_REACHEABLE = False
        # O índice 0 é onde o robô já está, então a busca começa no 1.
        for i in range(1, len(path)):
            if self.map_array[path[i]] == UNKNOWN:
                # Para alguns pixels antes da fronteira, mas nunca antes do início.
                return path[:max(1, i - self.frontier_margin_px)]

        self.GOAL_REACHEABLE = path[-1] == self.goal
        return path

    def segment_is_safe(self, a: tuple, b: tuple) -> bool:
        """True se a reta de a até b passa só por células livres, conhecidas e fora da zona de segurança."""
        n = int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2)) + 1
        rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return bool(np.all(self.map_array[rows, cols] == FREE) and np.all(~self.unsafe[rows, cols]))

    def simplify_path(self, path: list) -> list:
        """
        Simplifica o caminho removendo direções repetidas.

        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho simplificado.
        """
        if len(path) <= 2:
            return list(path)

        # 1) Remove pontos intermediários que seguem na mesma direção.
        waypoints = [path[0]]
        for prev, cur, nxt in zip(path, path[1:], path[2:]):
            if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
                waypoints.append(cur)
        waypoints.append(path[-1])

        # 2) Linha de visada: de cada ponto, pula para o mais distante alcançável em linha reta segura.
        simplified = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            j = len(waypoints) - 1
            while j > i + 1 and not self.segment_is_safe(waypoints[i], waypoints[j]):
                j -= 1
            simplified.append(waypoints[j])
            i = j
        return simplified

    def plot_path(self, path: list, simplified_path: list):
        """
        Exibe o mapa com o caminho completo e o simplificado.

        Args:
            path (list): O caminho completo encontrado.
            simplified_path (list): O caminho simplificado encontrado.
        """
        simplified_path = self.simplify_path(path)

        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='magenta', linewidth=1, label='Caminho Completo')
            simp_x, simp_y = zip(*simplified_path)
            plt.plot(simp_y, simp_x, color='red', linewidth=2, linestyle='--', label='Caminho Simplificado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        plt.show()

    def run(self, show_path=True, gif=None, frame_every=30):
        """
        Essa função é chamada pelo navegador para executar o algoritmo A* e gerar o caminho.
        Executa o processo completo: busca, reconstrução, simplificação e visualização do caminho.

        Args:
            show_path (bool): Se True, exibe o caminho graficamente.
            gif (str): Se informado, salva o GIF do processo de busca nesse arquivo.
            frame_every (int): Grava um frame a cada N nós expandidos (só vale com gif).

        Returns:
            list or None: Caminho simplificado ou None se não encontrado.
        """
        print("Iniciando busca pelo caminho...")
        came_from, final_node = self.find_path(record=gif is not None, frame_every=frame_every)

        if final_node:
            print("Reconstruindo caminho...")
            path = self.reconstruct_path(came_from, final_node)

            print("Robo não anda no disconhecido")
            path = self.know_path(path)

            print("Caminho encontrado, simplificando...")
            simplified_path = self.simplify_path(path)

            if gif:
                print("Salvando GIF...")
                self.save_gif(gif, path=path)   # troque por simplified_path se preferir

            print("Plotando o caminho...")
            if show_path:
                self.plot_path(path, simplified_path)

            return simplified_path
        else:
            print("Nenhum caminho pôde ser encontrado.")
            return None


def prep_map(map_path: str) -> np.array:
    """
    Prepara o mapa carregando e processando a imagem de entrada.

    Args:
        map_path (str): O caminho do arquivo do mapa.

    Returns:
        np.array: O mapa processado como um array numpy.
    """
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_array[map_array == 0] = 0
    map_array[map_array == 205] = 128
    map_array[map_array == 254] = 255
    map_array[(map_array >= 60) & (map_array != 128) & (map_array != 255)] = 0
    map_array = map_array.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    map_array = cv2.morphologyEx(map_array, cv2.MORPH_OPEN, kernel)
    map_array = np.flipud(map_array)
    map_array = np.pad(map_array, ((0, 200), (0, 200)), 'constant', constant_values=128)
    return map_array


def main():
    map_array = prep_map('map3.pgm')
    astar = AStarPathfinder(map_array, (60, 20), (60, 120), wall_influence=10.0, buffer_factor=3.0)
    astar.run(gif='astar.gif', frame_every=30)


if __name__ == '__main__':
    main()
Você é um engenheiro de robótica experiente em Python. Preciso completar o arquivo astar.py de um projeto da faculdade (IA e Robótica, Insper): path planning com A* para um robô real que anda num labirinto que ele só conhece em parte.

## Contexto do template
- Repositório: https://github.com/Insper/ia_robotica. Tem o astar.py (classe AStarPathfinder com os métodos vazios) e os mapas map1.pgm … map5.pgm.
- Os mapas têm 150x73 px e são "fotos" sucessivas do mesmo labirinto sendo explorado: o map1 é quase todo desconhecido, o map5 é bem mais conhecido.
- A função prep_map() já está pronta e NÃO deve ser alterada. Ela converte o PGM para: 0 = parede, 128 = desconhecido, 255 = livre. Também faz flipud e adiciona 200 px de borda desconhecida (128) à direita e embaixo.
- As coordenadas são (linha, coluna). O main() de exemplo usa map5.pgm, início (60, 20), objetivo (60, 120), wall_influence=10.0 e buffer_factor=3.0.
- O run() é chamado pelo software de navegação do robô. NÃO altere nomes de métodos, assinaturas existentes nem formatos de retorno: run() devolve o caminho simplificado (lista de tuplas (linha, coluna)) ou None; find_path() devolve (came_from, nó_final) ou (None, None). Pode adicionar parâmetros opcionais com valor padrão.

## Como o robô deve se comportar
1. Planeja um caminho do início até o objetivo, mesmo que o objetivo esteja na região desconhecida.
2. Só anda pela parte conhecida: o caminho devolvido termina na borda entre conhecido e desconhecido.
3. Lá ele para, o mapa é atualizado e o planejador é chamado de novo. Repete até chegar.
4. NUNCA pode encostar nem raspar na parede. Deve passar longe delas, mas ainda assim conseguir passar em corredores estreitos e com obstáculos extras.
5. Andar suave: poucos waypoints, sem zigue-zague nem curvas bruscas desnecessárias.

## O que implementar em cada método
- preprocess_map: garantir só 3 valores (0/128/255). Qualquer valor intermediário vira parede.
- create_potential_field: usar distance_transform_edt para calcular, em cada célula, a distância até a parede mais próxima (só paredes contam como obstáculo, não o desconhecido). Transformar isso num custo extra que cresce perto da parede, controlado por wall_influence (peso) e buffer_factor (alcance), por exemplo com decaimento exponencial. Adicionar um parâmetro robot_radius_px e marcar como intransponível toda célula com distância menor que ele. A ideia: o raio mínimo garante que o robô não bate; o custo suave faz ele preferir o meio do corredor sem bloquear passagens estreitas. Não inflar as paredes demais, senão os corredores estreitos somem.
- heuristic: distância euclidiana (admissível com 8 direções e custo diagonal √2, já que o custo extra do campo é sempre ≥ 0).
- find_path: A* com heapq, 8 vizinhos, custo do passo = comprimento do movimento (1 ou √2) + campo potencial da célula de destino. Proibir paredes, células abaixo do raio mínimo e diagonais que cortam quina. Células desconhecidas são permitidas (planejamento otimista), mas com um custo extra para o robô preferir o conhecido e não dar a volta por fora do labirinto pela borda de padding. Se o início estiver dentro da zona de segurança, deixar o robô sair dela em vez de falhar. Checar limites do array.
- reconstruct_path: seguir came_from do objetivo até o início e inverter.
- know_path: cortar o caminho no primeiro ponto que cai em célula desconhecida e recuar alguns pixels (parâmetro) para o robô parar com folga antes da fronteira. Se o caminho é todo conhecido, devolver inteiro. Definir self.GOAL_REACHEABLE = True só quando o caminho chega ao objetivo sem passar pelo desconhecido.
- simplify_path: primeiro remover pontos que seguem na mesma direção (como diz a docstring). Depois simplificar por linha de visada: de cada waypoint, pular para o ponto mais distante do caminho que dá para ligar em linha reta sem que nenhum pixel do segmento fique abaixo do raio mínimo de segurança. Manter o primeiro e o último ponto.

## Testes
Crie um test_maps.py que:
- Roda nos 5 mapas com início (60, 20) e objetivo (60, 120) e salva uma imagem por mapa (caminho completo, caminho simplificado e zona de segurança das paredes).
- Verifica com asserts: nenhum ponto do caminho completo nem dos segmentos simplificados cai em parede ou abaixo do raio mínimo; nenhum ponto do caminho devolvido está em célula desconhecida; o caminho começa no início.
- Simula a exploração: usa o ponto final do caminho no map1 como início no map2, e assim por diante até o map5.

## O que quero de volta
1. O astar.py completo.
2. O test_maps.py.
3. Uma explicação em português, função por função, das decisões tomadas (A*, heurística admissível, campo potencial, papel de wall_influence, buffer_factor e robot_radius_px, corte na fronteira, simplificação). Cada integrante da equipe precisa conseguir explicar o código numa arguição com o professor.
4. Um guia de ajuste para o robô físico: o que mudar se ele passar perto demais da parede, se não passar em corredor estreito, ou se fizer curvas bruscas. Avise que o raio em pixels depende da resolução do mapa (metros por pixel) e do tamanho do robô, e deixe claro onde eu ajusto isso.

Segue o astar.py original:
[cole aqui o conteúdo do astar.py]
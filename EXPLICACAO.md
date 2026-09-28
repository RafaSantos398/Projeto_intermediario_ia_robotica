# A* com campo potencial: explicação e guia de ajuste

## Visão geral

O robô recebe um mapa parcialmente explorado (0 = parede, 128 = desconhecido, 255 = livre) e precisa ir do início ao objetivo. O ciclo é:

1. O A* planeja o caminho inteiro até o objetivo, mesmo que o objetivo esteja no desconhecido (planejamento otimista).
2. O caminho é cortado antes da primeira célula desconhecida, porque o robô só anda no que conhece.
3. O caminho é simplificado em poucos waypoints retos.
4. O robô anda até o último waypoint, o mapa é atualizado e o `run()` é chamado de novo.

Nos 5 mapas do repositório o objetivo (60, 120) está sempre no desconhecido, então a resposta é sempre "ande até a fronteira". O `test_maps.py` simula essa exploração: o ponto de parada no map1 vira o início no map2, e assim por diante.

## Função por função

### `preprocess_map`
Garante que o mapa só tenha 0, 128 e 255. Qualquer valor intermediário vira **parede**. A escolha é conservadora: na dúvida, é melhor o robô desviar de algo que não existe do que bater em algo que existe.

### `create_potential_field`
É o coração da segurança. São três passos:

1. **Distância até a parede** (`distance_transform_edt`): para cada célula, a distância euclidiana em pixels até a parede mais próxima. Só a parede conta como obstáculo. O desconhecido não conta, senão o robô teria "medo" da fronteira e nunca exploraria.
2. **Zona de segurança** (`robot_radius_px`): toda célula com distância < `robot_radius_px` fica intransponível (`np.inf`). Isso garante que o centro do robô nunca chega perto o bastante para encostar na parede. É uma **garantia rígida**.
3. **Custo suave**: fora da zona, o custo extra de uma célula é

   ```
   custo(d) = wall_influence · exp(−(d − robot_radius_px) / buffer_factor)
   ```

   - Na borda da zona de segurança (d = raio) o custo vale `wall_influence`.
   - Ele cai exponencialmente com a distância. `buffer_factor` é o "alcance": a cada `buffer_factor` pixels o custo cai para ~37%.
   - Esse custo é uma **preferência**, não uma proibição: o robô prefere o meio do corredor, mas ainda passa num corredor estreito se esse for o único caminho (paga mais caro, mas passa).

**Por que separar raio rígido e custo suave?** Se toda a margem viesse do raio (paredes muito infladas), os corredores estreitos fechariam e o robô não acharia caminho. Com um raio pequeno, só o necessário para não bater, e um custo suave por cima, temos as duas coisas: segurança garantida e preferência pelo centro.

### `heuristic`
Usa a distância euclidiana. Ela é **admissível** (nunca superestima o custo real) porque:
- cada passo custa pelo menos o comprimento geométrico do movimento (1 ou √2), já que o campo potencial e o custo do desconhecido são ≥ 0;
- a distância euclidiana é o menor comprimento possível entre dois pontos.

Com heurística admissível, o A* garante o caminho de **menor custo**. A distância de Manhattan superestimaria em diagonais e não seria admissível com 8 vizinhos.

### `find_path` (o A*)
- **Fila de prioridade** (`heapq`) ordenada por `f = g + h`, onde `g` é o custo acumulado e `h` é a heurística.
- **8 vizinhos**. O custo do passo é o comprimento do movimento (1 ou √2) somado ao campo potencial da célula de destino, mais `unknown_cost` se a célula for desconhecida.
- **Proibições**: parede, célula dentro da zona de segurança, fora dos limites do array, e **diagonal que corta quina** (se uma das duas células ortogonais for parede, o movimento diagonal passaria "raspando" a quina).
- **Desconhecido permitido, mas caro** (`unknown_cost`): sem esse custo o A* sairia do labirinto por uma brecha e daria a volta pela borda de padding (200 px de desconhecido sem paredes). Com ele, o robô usa o máximo do conhecido e só entra no desconhecido quando precisa.
- **Início na zona de segurança**: se o robô já está colado na parede (erro de localização, mapa atualizado), ele pode andar por células inseguras, **desde que cada passo não o aproxime da parede**. Assim ele sai da zona em vez de o planejador falhar. A partir de uma célula segura, nunca se entra na zona.
- **Objetivo na zona de segurança**: é trocado pela célula segura mais próxima.
- Usa "lazy deletion": o mesmo nó pode entrar várias vezes no heap, e as cópias velhas são ignoradas pelo conjunto `closed`.

### `reconstruct_path`
Segue `came_from` do nó final até o início (que não tem predecessor) e inverte a lista.

### `know_path`
Percorre o caminho e acha o primeiro ponto em célula desconhecida. O caminho é cortado `frontier_margin_px` pontos **antes** dele, para o robô parar com folga, sem ficar exatamente na fronteira. O ponto 0 é ignorado porque o robô já está lá.
- Se nenhum ponto é desconhecido, o caminho inteiro é devolvido e `GOAL_REACHEABLE = True`: o objetivo é alcançável só pelo conhecido.
- Caso contrário, `GOAL_REACHEABLE = False`.

### `simplify_path`
São duas etapas:
1. **Direções repetidas**: remove pontos no meio de trechos retos (como pede a docstring original).
2. **Linha de visada**: a partir de cada waypoint, procura o waypoint **mais distante** que pode ser ligado por uma reta em que todos os pixels são livres, conhecidos e fora da zona de segurança (`segment_is_safe`). Isso elimina o zigue-zague da grade de 8 direções e deixa poucos waypoints com curvas suaves.

O primeiro e o último ponto sempre ficam. Se nenhuma reta for segura (por exemplo, o início está na zona de segurança), o algoritmo usa o próximo waypoint, que já está no caminho do A*.

## Testes (`test_maps.py`)
Rode com `python test_maps.py`. Imagens em `resultados/`: amarelo = zona de segurança, magenta = A* completo, vermelho = caminho simplificado.
- Roda os 5 mapas de (60, 20) até (60, 120).
- Verifica que nenhum ponto do A* nem dos segmentos simplificados está em parede ou abaixo do raio, que nenhum ponto devolvido está no desconhecido e que o caminho começa no início.
- Simula a exploração encadeada map1 → map5.
- Testa o início colado na parede (escape da zona de segurança).

## Guia de ajuste para o robô físico

Todos os parâmetros estão no construtor, em `astar.py:20-21`. O `main()` e o software de navegação passam `wall_influence` e `buffer_factor`. Os outros usam o valor padrão da assinatura, então é **ali que se muda** (ou passe no construtor, se tiver acesso a quem chama).

### ⚠️ O raio em pixels depende do mapa e do robô
`robot_radius_px` está em **pixels**, não em metros:

```
robot_radius_px = (raio_do_robô_m + folga_m) / resolução_m_por_pixel
```

A resolução vem do `.yaml` do mapa (campo `resolution`, por exemplo `0.05` m/px). Exemplo: robô com raio de 0,10 m, folga de 0,05 m e 0,05 m/px dá 3 px, que é o padrão atual. **Se a resolução ou o robô mudar, esse número muda.** Meçam o robô de verdade (raio do círculo que o envolve, a partir do centro de rotação).

Um corredor só é transitável se tiver pelo menos `2 · robot_radius_px + 1` pixels livres de largura.

| Sintoma | O que mudar |
|---|---|
| **Passa perto demais / raspa na parede** | 1º: aumente `robot_radius_px` (é a garantia rígida). 2º: aumente `wall_influence` (mais força para o centro) ou `buffer_factor` (a repulsão alcança mais longe). Aumente também a folga se a localização (AMCL) for imprecisa. |
| **Não passa em corredor estreito / "caminho não encontrado"** | Diminua `robot_radius_px` (sem ficar abaixo do raio físico real em pixels). Diminua `wall_influence` se ele achar caminho mas preferir uma volta enorme. `buffer_factor` não bloqueia passagens, só as torna caras. |
| **Curvas bruscas / muitos waypoints** | Aumentar `robot_radius_px` também afasta as retas da parede. Se o problema for a quina, `buffer_factor` maior faz o A* abrir mais a curva. Curvas no *waypoint* em si são controladas pelo seguidor de caminho (controlador), não pelo A*. |
| **Sai do labirinto pela borda / explora por fora** | Aumente `unknown_cost`. |
| **Não entra no desconhecido quando deveria / volta muito para usar o conhecido** | Diminua `unknown_cost`. |
| **Para longe demais / perto demais da fronteira** | Ajuste `frontier_margin_px`. |

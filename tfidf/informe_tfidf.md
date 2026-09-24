# Informe: Similitud de documentos con TF-IDF y comparación con Bag-of-Words (PyTorch)

## Metodología

1. **Lectura y preprocesamiento**: se reutilizan las funciones de
   `bow/bow_similarity.py` (minúsculas, eliminación de acentos, de caracteres
   no alfabéticos y de stopwords), de modo que ambos modelos comparten el
   mismo vocabulario de **408 palabras**.
2. **TF**: frecuencia relativa de cada palabra en el documento (la fila de la
   matriz BoW dividida entre el número de tokens del documento).
3. **IDF**: se usa la variante suavizada
   `idf(t) = log((1 + N) / (1 + df(t))) + 1`, con `N = 12` documentos y
   `df(t)` el número de documentos que contienen `t`. El suavizado evita
   divisiones por cero y que una palabra presente en todos los documentos
   reciba peso nulo.
4. **TF-IDF**: producto `TF × IDF`, lo que da un tensor de PyTorch `(12, 408)`.
5. **Similitud del coseno**: se calcula con la misma función que en la parte 1
   (`torch.nn.functional.cosine_similarity`), lo que da una matriz `(12, 12)`.

Código: `tfidf_similarity.py`.

Las palabras con menor IDF, que TF-IDF penaliza más, son las más repartidas
por el corpus: *movil* (1.49), *tecnologia*, *electricos*, *usuarios*,
*mundo*, *nueva*, *smartphones* (1.77), *ademas*, *coche*, *electrica*
(1.96). Una palabra que aparece en un solo documento tiene IDF = 2.87.

## Hallazgos principales

**Pares más similares:**

| Par | TF-IDF | BoW | Rango TF-IDF (BoW) |
|---|---|---|---|
| doc09 - doc12 | 0.226 | 0.328 | 1 (1) |
| doc11 - doc12 | 0.199 | 0.304 | 2 (2) |
| doc04 - doc12 | 0.166 | 0.251 | 3 (3) |
| doc04 - doc09 | 0.133 | 0.219 | 4 (4) |
| doc05 - doc09 | 0.129 | 0.208 | 5 (5) |
| doc02 - doc11 | 0.126 | 0.203 | 6 (6) |
| doc05 - doc08 | 0.123 | 0.191 | 7 (7) |
| doc09 - doc11 | 0.118 | 0.167 | 8 (9) |
| doc01 - doc06 | 0.090 | 0.138 | 9 (12) |
| doc07 - doc11 | 0.090 | 0.190 | 10 (8) |

**Pares menos similares** (similitud = 0.000): los mismos que con BoW
(doc02-doc09, doc02-doc05, doc02-doc04, doc01-doc09, doc01-doc03, entre
otros). Si dos documentos no comparten ninguna palabra, la similitud es 0 con
cualquier ponderación.

**Similitud media por grupos temáticos** (pares de documentos distintos):

| Grupos | BoW | TF-IDF |
|---|---|---|
| móvil - móvil | 0.081 | 0.046 |
| VE - VE | 0.156 | 0.097 |
| móvil - VE | 0.025 | 0.017 |
| híbrido - móvil | 0.112 | 0.059 |
| híbrido - VE | 0.148 | 0.100 |
| **Media global** | **0.085** | **0.051** |

## Análisis y comparación con BoW

- **Todas las similitudes bajan con TF-IDF** (la media pasa de 0.085 a
  0.051). Ningún par sube: la matriz de diferencias TF-IDF − BoW es ≤ 0 en
  todas las celdas. Esto es lo esperable: los términos que comparten varios
  documentos son justamente los de IDF bajo, así que pierden peso frente a
  las palabras exclusivas de cada documento, que no aportan nada al producto
  escalar.
- **La ordenación de los pares apenas cambia.** Los 7 primeros pares son
  idénticos y en el mismo orden en ambos modelos, y los pares con similitud
  0 no cambian. La estructura temática que se vio con BoW se mantiene: el
  clúster de vehículos eléctricos es el más cohesionado, los documentos
  híbridos (doc11, doc12) siguen haciendo de puente, y los pares de temas
  disjuntos (móvil - VE) siguen siendo los menos similares.
- **Los cambios de rango se explican por el peso de palabras genéricas:**
  - *doc07 - doc11* baja del puesto 8 al 10 (0.190 → 0.090, la mayor
    caída relativa del top 10). Con BoW la similitud dependía casi por
    completo de *movil*, que aporta 0.109 de los 0.190. Al ser la palabra
    con menor IDF del corpus, su aportación cae a 0.038.
  - *doc01 - doc06* sube del puesto 12 al 9. Su similitud se debe sobre
    todo a *pantalla*, un término más específico que conserva casi todo su
    peso (0.046 → 0.040), mientras que la aportación de *movil* se reduce
    (0.031 → 0.010).
  - *doc09 - doc12*, el par más similar en ambos modelos, se apoya en
    *carga*, *vehiculos*, *electricos* y *conductores*: comparten un subtema
    concreto (la recarga de vehículos eléctricos) y no solo vocabulario
    genérico.
- **Contraste entre temas:** la relación entre la similitud intra-tema y la
  inter-tema (móvil - VE) es parecida en ambos modelos: ≈3.2× (BoW) frente a
  ≈2.7× (TF-IDF) dentro del clúster móvil, y ≈6.2× frente a ≈5.7× dentro
  del clúster VE. En este corpus TF-IDF no separa mejor los temas: las
  palabras que definen cada tema (*movil*, *smartphones*, *electricos*,
  *coche*) aparecen en muchos documentos, así que TF-IDF también las
  penaliza. Lo que sí hace es dar más peso a las coincidencias en subtemas
  específicos (*pantalla*, *carga*) frente a las coincidencias en términos
  generales.
- **Limitaciones:** con solo 12 documentos cortos, el IDF tiene poca
  resolución (df va de 1 a unos pocos documentos) y muchos pares no
  comparten ninguna palabra. Con un corpus más grande o con lematización
  (p. ej. unificar *electrico / electricos / electrica*), las diferencias
  entre ambos modelos probablemente serían mayores.

## Conclusión

TF-IDF confirma los resultados de BoW: los documentos del mismo tema, y
sobre todo los que comparten un subtema concreto, son los más similares, y
los de temas disjuntos tienen similitud prácticamente nula. La diferencia
principal es que TF-IDF reduce el efecto de las palabras genéricas del
corpus. Así, la similitud se basa en el vocabulario específico que comparten
los documentos y no en términos que aparecen en casi todos, aunque en este
corpus pequeño eso no aumenta la separación entre los dos temas.

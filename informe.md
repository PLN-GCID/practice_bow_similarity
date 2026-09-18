# Informe: Similitud de documentos con Bag-of-Words y similitud del coseno (PyTorch)

## Metodología

1. **Lectura**: los 12 documentos (`doc01.docx`...`doc12.docx`) se leyeron con `python-docx`.
2. **Preprocesamiento**: minúsculas, eliminación de acentos, eliminación de
   caracteres no alfabéticos y de stopwords en español (lista propia, sin
   dependencias externas).
3. **Bag-of-Words**: se construyó un vocabulario global de **408 palabras**
   únicas y se representó cada documento como un vector de frecuencias sobre
   ese vocabulario, usando un tensor de PyTorch `(12, 408)`.
4. **Similitud del coseno**: se calculó con `torch.nn.functional.cosine_similarity`
   entre todos los pares de documentos, obteniendo una matriz `(12, 12)`.

Código: `bow_similarity.py`.

## Clasificación temática observada de los documentos

| Documento | Tema |
|---|---|
| doc01 | Móvil (smartphone gama alta) |
| doc02 | Móvil (batería) |
| doc03 | Móvil (apps más descargadas) |
| doc04 | Vehículo eléctrico (futuro del sector) |
| doc05 | Vehículo eléctrico (subvenciones) |
| doc06 | Móvil (pantallas plegables) |
| doc07 | Móvil (realidad aumentada) |
| doc08 | Vehículo eléctrico (modelo económico) |
| doc09 | Vehículo eléctrico (supercargadores) |
| doc10 | Móvil (seguridad) |
| doc11 | **Híbrido**: smartphones integrados en vehículos eléctricos |
| doc12 | **Híbrido**: app móvil para carga de vehículos eléctricos |

## Hallazgos principales

**Pares más similares:**

| Par | Similitud coseno |
|---|---|
| doc09 - doc12 | 0.328 |
| doc11 - doc12 | 0.304 |
| doc04 - doc12 | 0.251 |
| doc04 - doc09 | 0.219 |
| doc05 - doc09 | 0.208 |

**Pares menos similares** (similitud = 0.000): doc02-doc09, doc02-doc05,
doc02-doc04, doc01-doc09, doc01-doc03, entre otros.

## Análisis

- Los documentos que comparten temática tienden a ser más similares entre sí:
  los pares más altos son casi todos dentro del clúster de vehículos
  eléctricos (doc04, doc05, doc08, doc09) o involucran a los documentos
  híbridos doc11 y doc12, que comparten vocabulario ("smartphone", "app",
  "móvil") con el clúster de tecnología móvil y ("coche eléctrico", "carga",
  "batería del vehículo") con el clúster de VE.
- Los documentos híbridos (doc11, doc12) actúan como **puente** entre los dos
  temas: tienen similitud moderada tanto con documentos puramente móviles
  como con documentos puramente de vehículos eléctricos, mientras que un
  documento puro de un tema tiene similitud prácticamente nula con un
  documento puro del otro tema (p. ej. doc01 "smartphone gama alta" vs.
  doc09 "supercargadores" = 0.000).
- Dentro del propio clúster móvil las similitudes son relativamente bajas
  (0.0–0.2) porque cada artículo cubre un subtema distinto (batería, apps,
  pantallas plegables, realidad aumentada, seguridad) con poco vocabulario
  compartido más allá de términos genéricos como "smartphone" o "móvil".
- En conjunto, los resultados confirman la hipótesis esperada: **la similitud
  del coseno sobre representaciones bag-of-words refleja razonablemente bien
  la cercanía temática entre documentos**, siendo mayor entre documentos del
  mismo tema (o que comparten un subtema muy específico, como la carga de
  vehículos eléctricos) y prácticamente nula entre documentos de temas
  disjuntos.

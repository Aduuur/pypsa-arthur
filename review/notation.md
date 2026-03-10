# Notation zur ARO-/Robust-Formulierung

## Ziel dieser Datei

Diese Datei sammelt die Notation, damit die Gleichungen aus [`aro_review.md`](./aro_review.md) einheitlich referenziert werden koennen.

## Grundidee

Es gibt:

- eine gemeinsame First-Stage-Entscheidung fuer Investitionen
- pro Szenario eigene Second-Stage-Entscheidungen fuer den Betrieb
- eine Worst-Case-Epigraphvariable

Die Szenarien repraesentieren hier diskrete `cutouts`, also alternative Wetter- und Nachfragepfade.

## Mengen

- `S`: Menge der diskreten Szenarien
- `T_s`: Zeitschritte innerhalb des Szenarios `s`
- `B`: Menge der Busse bzw. Knoten
- `G`: Menge der Generatoren
- `L`: Menge der Links
- `U`: Menge der `StorageUnits`
- `R`: Menge der `Stores`
- `K^{GC}`: Menge der zeitaggregierten GlobalConstraints

Ergaenzend:

$$ S = \{1, \dots, N_S\} $$

$$ T_s = \{1, \dots, N_T^{(s)}\} $$

## Parameter

- `w_{t,s}`: Snapshot-Gewicht des Zeitschritts `t` im Szenario `s`
- `c_i^{inv}`: jaehrlicher Investitionskostensatz des investierbaren Assets `i`
- `c_g^{var}`: Marginalkosten des Generators `g`
- `c_l^{var}`: Marginalkosten oder sonstige lineare Betriebskosten des Links `l`
- `c^{LS}`: Load-Shedding-Strafkosten
- `\bar p_{g,t,s}`: Verfuegbarkeitsfaktor oder obere relative Produktionsgrenze des Generators `g`
- `\eta_u^{ch}, \eta_u^{dis}`: Lade- und Entladewirkungsgrade der `StorageUnit` `u`
- `\lambda_u, \lambda_r`: stehende Verluste von `StorageUnit` `u` bzw. `Store` `r`
- `inflow_{u,t,s}`: exogener Zufluss in `StorageUnit` `u`
- `\alpha_{g,k}`: Beitrag des Generators `g` zur GlobalConstraint `k`
- `\beta_{l,k}`: Beitrag des Links `l` zur GlobalConstraint `k`
- `B_k`: urspruengliche rechte Seite der GlobalConstraint `k`
- `a`: `annual_scale`

## Entscheidungsvariablen

### First Stage

- `x_i >= 0`: gemeinsame Investitionsvariable des Assets `i`
- Beispiele: `x_g`, `x_l`, `x_u^{pow}`, `x_r^{ene}`

### Second Stage je Szenario

- `p_{g,t,s}`: Generatorleistung
- `f_{l,t,s}`: Linkfluss
- `p^{ch}_{u,t,s}`, `p^{dis}_{u,t,s}`: Lade- und Entladeleistung
- `soc_{u,t,s}`: Ladezustand einer `StorageUnit`
- `p_{r,t,s}`: Store-Leistung
- `e_{r,t,s}`: Energieinhalt eines `Store`
- `shed_{b,t,s}`: abgeworfene Last an Bus `b`

### Worst-Case-Variable

- `z`: Epigraphvariable fuer die maximalen Betriebskosten ueber alle aktiven Szenarien

## Zielfunktion

$$ \min_{x, y, z} \sum_i c_i^{inv} x_i + z $$

mit `y = {y_s}_{s \in S}` als Sammelbezeichnung aller Second-Stage-Variablen.

## Operative Kosten pro Szenario

$$ C_s^{op}(x, y_s) = \sum_{t \in T_s} w_{t,s} \left[ \sum_{g \in G} c_g^{var} p_{g,t,s} + \sum_{l \in L} c_l^{var} f_{l,t,s} + \sum_{b \in B} c^{LS} shed_{b,t,s} + C_{t,s}^{other} \right] $$

Dabei steht `C_{t,s}^{other}` fuer weitere lineare Betriebs- oder CO2-Kosten, die der Code in den linearen Betriebskostenausdruck integriert.

## Worst-Case-Epigraphbedingungen

Fuer alle aktiven Master-Szenarien `s in S_k` gilt:

$$ z \ge C_s^{op}(x, y_s) $$

Das ist die lineare Epigraph-Darstellung des Worst Case:

$$ z = \max_{s \in S_k} C_s^{op}(x, y_s) $$

im Optimum.

## Technische Kernnebenbedingungen

### Generatorgrenzen

$$ 0 \le p_{g,t,s} \le \bar p_{g,t,s} \cdot x_g $$

falls `g` investierbar ist.

### Linkgrenzen

$$ -x_l^- \le f_{l,t,s} \le x_l^+ $$

oder aequivalent komponentenspezifisch, abhaengig von der konkreten PyPSA-Parametrisierung.

### Nodale Bilanz

Fuer jeden Bus `b`, Zeitpunkt `t`, Szenario `s` gilt typischerweise:

$$ \sum_{g \in G(b)} p_{g,t,s} + \sum_{l \in L^{in}(b)} f_{l,t,s} - \sum_{l \in L^{out}(b)} f_{l,t,s} + \sum_{u \in U^{dis}(b)} p^{dis}_{u,t,s} - \sum_{u \in U^{ch}(b)} p^{ch}_{u,t,s} + shed_{b,t,s} = d_{b,t,s} $$

Die genaue Summenstruktur ist komponentenabhaengig, aber dies ist die typische kompakte Form.

### StorageUnit-Dynamik

$$ soc_{u,t,s} = (1-\lambda_u) soc_{u,t-1,s} + \eta_u^{ch} p^{ch}_{u,t,s} - \frac{1}{\eta_u^{dis}} p^{dis}_{u,t,s} + inflow_{u,t,s} $$

### Store-Dynamik

$$ e_{r,t,s} = (1-\lambda_r)e_{r,t-1,s} - p_{r,t,s} $$

Vorzeichenkonventionen koennen je nach Komponente differieren; die lineare Bilanzstruktur bleibt aber identisch.

## Szenarioeigene zyklische Schliessung

Fuer zyklische Speicherelemente gilt:

$$ soc_{u,t^{first}_s,s} = soc_{u,t^{last}_s,s} \quad \forall u \in U^{cyc}, \forall s \in S $$

$$ e_{r,t^{first}_s,s} = e_{r,t^{last}_s,s} \quad \forall r \in R^{cyc}, \forall s \in S $$

Diese Bedingungen sind methodisch zentral, weil sie jede Witterungs- und Nachfragebahn fuer sich schliessen.

### Warum diese Schliessung in diesem Projekt wichtig ist

Beim Stapeln der Szenarien in einer gemeinsamen Snapshot-Liste entstuende sonst faelschlich:

$$ soc_{u,1,s} = (1-\lambda_u) soc_{u,T,s-1} + \dots $$

Das wuerde Energie ueber Szenariogrenzen transportieren.

Die Implementierung entfernt genau diese falschen Grenzzeilen und ersetzt sie durch die szenarioeigene Zyklisierung.

## Per-Szenario-GlobalConstraints

Fuer jede zeitaggregierte GlobalConstraint `k in K^{GC}` mit Vergleichsoperator `\odot_k` gilt:

$$ \sum_{t \in T_s} w_{t,s} \left( \sum_{g \in G} \alpha_{g,k} p_{g,t,s} + \sum_{l \in L} \beta_{l,k} f_{l,t,s} \right) \odot_k \frac{B_k}{a} \quad \forall s \in S $$

mit `\odot_k in {<=, =, >=}`.

Der Faktor `a` korrigiert die Skalierung der Snapshot-Gewichte auf Jahreskosten bzw. Jahresbudgets.

## Robustes Grundproblem

Ohne Outer Loop laesst sich das robuste Masterproblem kompakt schreiben als:

$$ \min_{x, z, \{y_s\}_{s \in S_k}} \sum_i c_i^{inv} x_i + z $$

unter

$$ z \ge C_s^{op}(x, y_s) \quad \forall s \in S_k $$

$$ y_s \in Y_s(x) \quad \forall s \in S_k $$

$$ x \in X $$

Hier ist `X` die Menge der zulaessigen Investitionsentscheidungen und `Y_s(x)` die Menge der zulaessigen Dispatchentscheidungen im Szenario `s` bei gegebenem Portfolio `x`.

## Outer Loop in `solve_aro.py`

Das ARO-Screening arbeitet mit `S_k subseteq S` als aktueller Master-Szenariomenge.

Nach Loesung des Masters wird das aktuelle Portfolio `x_k` auf allen Szenarien bewertet:

$$ \hat C_s(x_k) = \min_{y_s \in Y_s(x_k)} C_s^{op}(x_k, y_s) \quad \forall s \in S $$

Das schlechteste Szenario ist:

$$ s_k^\star \in \arg\max_{s \in S} \hat C_s(x_k) $$

### Klassischer exakter Update-Schritt

Ohne Rotation gilt:

$$ S_{k+1} = S_k \cup \{s_k^\star\} $$

### Heuristischer Update-Schritt mit FIFO-Fenster

Mit Fensterlaenge `K` gilt:

$$ S_{k+1} = \text{FIFO\_window}(S_k \cup \{s_k^\star\}, K) $$

Damit kann ein frueheres kritisches Szenario spaeter wieder aus dem Master verschwinden.

## Gap-Definition im Code

Die Implementierung verwendet:

$$ gap_k = \frac{\max_{s \in S} \hat C_s(x_k) - \max_{s \in S_k} \hat C_s(x_k)}{\max(1, |\max_{s \in S} \hat C_s(x_k)|)} $$

Diese Groesse misst, wie viel schlechter der globale Worst Case gegenueber dem schlimmsten bereits im Master enthaltenen Szenario noch ist.

## Interpretationshilfe

- `gap_k = 0`: das globale schlechteste Szenario liegt bereits auf dem Kostenlevel des Master-Worst-Case
- kleines `gap_k`: das aktuelle Fenster repraesentiert die kritische Szenariomenge praktisch gut
- grosses `gap_k`: ausserhalb des Masters existiert noch deutlich haertere Stresslage

## Terminologieempfehlung

Die mathematisch sauberste Bezeichnung fuer das Projekt lautet:

"diskrete zweistufige robuste Optimierung mit szenarioabhaengigem Recourse und optionalem Rolling-Window-Szenario-Screening."

`ARO` ist als Projektlabel verstaendlich, sollte aber in der schriftlichen Beschreibung praezisiert werden:

- ohne Rotation: exaktes endliches robustes Mehrszenario-Modell
- mit Rotation: heuristische Approximation des exakten Modells

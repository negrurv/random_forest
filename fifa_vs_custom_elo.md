# FIFA Rankings vs. Custom Elo Model Comparison (June 2026)

This document compares the dynamically calculated point-in-time Elo ratings from our quantitative research pipeline with the official FIFA/Coca-Cola Men's World Rankings (last published June 11, 2026).

---

## 📊 Side-by-Side Comparison

Below is a comparison of the top teams in both systems as of **June 28, 2026**:

| Country | Custom Elo Rank | Custom Elo Points | Official FIFA Rank | Official FIFA Points | Discrepancy (Rank Diff) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Argentina** | 1 | 2217.0 | 1 | 1889.06 | 0 |
| **Spain** | 2 | 2194.0 | 3 | 1856.03 | **+1** |
| **France** | 3 | 2175.0 | 2 | 1887.11 | **-1** |
| **England** | 4 | 2096.0 | 4 | 1847.68 | 0 |
| **Brazil** | 5 | 2087.0 | 5 | 1772.01 | 0 |
| **Colombia** | 6 | 2078.0 | 13 | 1698.35 | **+7** |
| **Netherlands** | 7 | 2040.0 | 8 | 1749.20 | **+1** |
| **Portugal** | 8 | 2036.0 | 7 | 1755.09 | **-1** |
| **Morocco** | 9 | 2018.0 | 6 | 1769.98 | **-3** |
| **Germany** | 10 | 1992.0 | 9 | 1743.54 | **-1** |
| **Mexico** | 11 | 1990.0 | 15 | 1650.00* | **+4** |
| **Japan** | 12 | 1989.0 | 18 | 1628.00* | **+6** |
| **Belgium** | 13 | 1967.0 | 10 | 1733.93 | **-3** |
| **Norway** | 14 | 1964.0 | 31 | 1600.00* | **+17** |

*\*Note: Official points for lower ranks are approximated based on recent June 2026 data. Norway's official FIFA rank is 31st; Mexico is 15th; Japan is 18th.*

---

## 🔍 Key Discrepancies Explained

### 1. Colombia (FIFA #13 vs. Custom #6)
In the official FIFA rankings, Colombia sits at 13th, while our model places them 6th (right behind England).
* **The Reason:** Our custom model heavily rewards Colombia's incredibly high-margin wins. Over their last 32 matches, Colombia won 24 times, including dominant scorelines like **5-1 against the USA** and **6-0 against Venezuela**. Our custom model multiplies Elo changes by goal differential, whereas the official FIFA formula ignores scorelines.
* **Friendly Weighting:** Colombia won several high-profile friendly matches (e.g., beating Spain 1-0 and beating Mexico 4-2). FIFA rankings assign friendlies a very low coefficient ($I=10$), whereas our model scales them at $K=20$.

### 2. Norway (FIFA #31 vs. Custom #10)
Norway is ranked 31st officially but slides into the top 10 in our custom model.
* **The Reason:** Despite failing to qualify for Euro 2024, Norway's individual match results in qualifiers, friendlies, and Nations League have been mathematically strong (large margins). In the 2026 World Cup group stage, they won their matches against Senegal (3-2) and Iraq (4-1). Because they have elite players (Haaland, Ødegaard) and win by high margins when they do win, our goal differential multipliers have inflated their ranking.

### 3. Belgium (FIFA #10 vs. Custom #17)
Belgium sits in the official FIFA top 10 but drops to 17th in our custom Elo.
* **The Reason:** FIFA's ranking formula is **sticky** because it was seeded in 2018 using legacy points. Belgium held the #1 ranking for years and has decayed slowly despite disappointing tournaments (2022 World Cup group exit, Euro 2024 round of 16 exit). Our model has no legacy seeding and recalculates pure performance from 1970 onwards, reacting much faster to Belgium's decline.

---

## 🧮 Mathematical Formula Comparison

The differences between the two systems are rooted in their mathematical formulations:

### 1. Goal Differential Multiplier ($G$)
* **Custom Elo Model:** Multiplies the rating shift ($\Delta$) by a scoreline margin factor:
  $$G = \begin{cases} 1.0 & \text{if } \text{GD} \le 1 \\ 1.5 & \text{if } \text{GD} = 2 \\ \frac{11 + \text{GD}}{8} & \text{if } \text{GD} \ge 3 \end{cases}$$
* **FIFA World Ranking:** Does **not** include goal differential. A 1-0 win and a 6-0 win yield the exact same rating shift.

### 2. Rating Horizon and Seeding
* **Custom Elo Model:** Simulates all matches chronologically from **1970 onwards**, with all teams starting equally at **1500**. This ensures that ratings reflect actual cumulative performance without historical bias.
* **FIFA World Ranking:** Adopted its Elo-based formula ("SUM algorithm") in **August 2018**, but seeded teams based on their position in the old, flawed ranking system. This carried over decades of legacy points.

### 3. Match Weights ($K$-factor vs. Importance $I$)
* **Custom Elo Model:** Uses standard $K$-factors:
  * Friendlies: $20$
  * Qualifiers: $40$
  * Major continental tournaments: $50$
  * World Cup matches: $60$
* **FIFA World Ranking:** Uses importance weights ($I$):
  * Friendlies outside international calendar: $5$
  * Friendlies inside international calendar: $10$
  * Nations League / Qualifiers: $15$ - $25$
  * Continental final tournaments: $35$ - $40$
  * World Cup matches: $50$ - $60$
  * By making friendlies so weak ($I=10$), FIFA prevents friendly-heavy schedules from shifting rankings, but it also lags behind when a team undergoes a real surge in quality.

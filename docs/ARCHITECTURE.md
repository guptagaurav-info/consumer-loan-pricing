# How this project works

Colour key: **blue** = external data · **lavender** = code step · **red** = check, gate or decision · **green** = good outcome · **yellow** = caution · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    A["SOURCE: LendingClub issued loans 2007–2018Q4<br/>2.26M loans · copy on Hugging Face (CC0)"]
    B["SOURCE: daily 3-year Treasury yield<br/>US Treasury via defeatbeta (ODC-BY)"]
    I{"ingest.py<br/>sha256 matches config?"}
    A --> I
    B --> I
    I -- "no" --> X["Stop: source changed"]
    I -- "yes" --> P
    subgraph P["Prepare · pipeline.py"]
      P1["36-month loans, credit-policy loans only"]
      P2{"Leakage guard: model fields must be<br/>known at approval (tested)"}
      P3["Complete vintages: issued to Dec 2015<br/>(term ended before the Mar 2019 data cut)"]
    end
    subgraph R["Tests"]
      R1["1. Priced vs realised: pool IRR after 1% servicing,<br/>by sub-grade, grade and vintage year"]
      R2["2. Vintage curves: share of $ that stopped paying,<br/>by month on book"]
      R3["3. Early read at month 6/12/18/24 vs two baselines,<br/>using only data available at the time"]
      R4["4. Out-of-time screen: train 2010–13, tune 2014, test 2015;<br/>grade-only vs linear vs gradient boosting"]
    end
    O["reports/metrics.json · CSVs · figures/*.png"]
    P --> R --> O
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef step fill:#ece8ff,stroke:#6b5fbf,color:#1d1640
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef out fill:#eeeeee,stroke:#777,color:#222
    class A,B src
    class P1,P3,R1,R2,R3,R4 step
    class I,X,P2 gate
    class O out
```

![Pipeline](../reports/figures/flow-1.png)

## 2. How a credit committee would run a consumer loan book with these results

```mermaid
flowchart TD
    S["New vintage purchased"]
    E["Expected lifetime loss set at pricing<br/>(sub-grade mix × seasoned history)"]
    M{"Month-12 and month-18 read:<br/>projected lifetime loss above the pricing assumption?"}
    K["Keep buying at current price"]
    W["Gap above 1 point at month 12"]
    A["Act before month 18: raise coupons or cut the<br/>riskiest grades; re-price forward-flow terms"]
    G{"Realised return of the lowest grades<br/>below the safest grade?"}
    C["Cap or stop E–G purchases;<br/>screening inside a grade cannot fix price"]
    Z["Within-grade screen (skip riskiest 10–20%)<br/>+ monthly calibration check, not only PSI"]
    S --> E --> M
    M -- "no" --> K --> Z
    M -- "yes" --> W --> A --> G
    G -- "yes" --> C --> Z
    G -- "no" --> Z
    classDef step fill:#ece8ff,stroke:#6b5fbf,color:#1d1640
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef good fill:#dff5e3,stroke:#2e8b57,color:#0d3318
    classDef warn fill:#fff4d6,stroke:#c9a227,color:#3d2f00
    class S,E,Z step
    class M,G gate
    class K good
    class W,A,C warn
```

![Decision flow](../reports/figures/flow-2.png)

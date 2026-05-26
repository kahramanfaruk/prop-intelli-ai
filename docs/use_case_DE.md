# PropIntelli AI — Use Case (Kurzfassung)

## Problem

Immobilien-Exposés (z. B. ImmoScout-Exports) kommen als PDFs in sehr
unterschiedlichen Layouts. Dieselben Informationen — **Preis, Wohnfläche, Lage,
Ausstattung, Baujahr** — stehen je nach Anbieter unter anderen Bezeichnungen,
teils als Tabelle, teils als Fließtext, teils als Scan. Heute werden diese Daten
**manuell abgetippt** und in CRM oder Excel übertragen: langsam, teuer,
fehleranfällig und der Engpass in der Lead-Verarbeitung.

## Lösung

Eine KI-gestützte Pipeline, die aus unstrukturierten Exposés ein **validiertes,
strukturiertes Datenmodell** erzeugt — mit Konfidenzwerten, Plausibilitätsprüfung
und einer **Human-in-the-Loop-Steuerung**: sichere Ergebnisse laufen automatisch
durch, unsichere werden gezielt zur Prüfung vorgelegt.

- **Bronze → Silver → Gold** (Medallion-Architektur)
- **Hybride Extraktion**: deterministische Regeln (offline, schnell, prüfbar) +
  optionales LLM (Ollama / OpenAI / Azure OpenAI) für Freitext und Sonderfälle
- **Datenqualität**: Pflichtfelder, Wertebereiche, Plausibilität (z. B. €/m²)
- **Strukturierte Ablage**: SQLite (Silver) und DuckDB/Parquet (Gold)

## Zielgruppe

Immobilien-Analyst:innen, PropTech-Plattformen, Investment-/Bewertungsteams und
CRM-Systeme, die Exposés automatisiert und konsistent erfassen wollen.

## Geschäftsnutzen

- **Wegfall der manuellen Erfassung** — Prüfen statt Abtippen.
- **Schnellere Lead-Verarbeitung** — Auto-Freigabe für sichere Fälle.
- **Konsistente Datenqualität** — fehlerhafte Werte werden vor dem CRM markiert.
- **Nachvollziehbar & lernend** — jeder Lauf wird auditiert; menschliche
  Korrekturen fließen als verlässliche Labels zurück.

## Erfolgsmetriken (gemessen am Test-Korpus)

| Metrik | Wert |
| --- | --- |
| Feldgenauigkeit | **100 %** |
| Macro-F1 | **0,995** |
| Exact-Match-Quote | **100 %** |

Gemessen mit dem Evaluations-Harness gegen maschinenlesbare Ground-Truth — allein
mit der deterministischen Basis (ohne LLM, vollständig offline).

## Azure-Bezug (Produktion)

Lokale Bausteine bilden 1:1 auf Azure ab: Blob Storage (Bronze), Azure SQL /
Microsoft Fabric Lakehouse (Silver/Gold), Azure OpenAI & Document Intelligence
(KI/OCR), Container Apps / Functions (Services) und Application Insights
(Monitoring).

> **Kernbotschaft:** Das ist kein PDF-Parser, sondern eine skalierbare
> Document-Intelligence-Plattform mit Feedback-Schleife, Evaluation und
> Enterprise-Integration.

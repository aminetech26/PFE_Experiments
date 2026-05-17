# Jetson Nano GUI PRD

## Problem Statement

The PV-FDD project needs a Jetson Nano deployable GUI that demonstrates the trained fault detection and diagnosis pipeline in a realistic edge setting. The current ML/DL work produces anomaly scores, fault diagnosis predictions, thresholds, calibration artifacts, and model metadata, but there is no operator-facing interface that turns these artifacts into a practical monitoring workflow.

The GUI must support a thesis/demo deployment on a constrained Jetson Nano 4GB device. It should be operator-first, replay-based for v1, locally accessible only on the Jetson, and structured like a production-like prototype rather than a research notebook.

## Solution

Build a local web UI deployed on the Jetson Nano in kiosk mode. The Jetson runs a lightweight Python backend and opens a local browser dashboard at startup. The first version supports replaying recorded PV data, running Task A anomaly detection and Task B fault diagnosis, displaying alerts, managing incidents, exporting PDF reports, monitoring drift/distribution shift, and administering model artifacts/settings.

The GUI has two user modes:

- Operator Mode: monitoring, alert acknowledgement, alert details, history, PDF export.
- Admin Mode: PIN-protected artifact management, thresholds/calibration, replay input configuration, diagnostics, logs, restart/shutdown controls.

Access is local-only:

```text
Backend binds to 127.0.0.1
Chromium opens the dashboard in kiosk mode
No LAN/cloud access in v1
```

## User Stories

1. As an operator, I want the dashboard to open automatically when the Jetson boots, so that I can monitor the PV system without using a terminal.
2. As an operator, I want to see a clear system status, so that I immediately know whether the PV system is normal, warning, or faulty.
3. As an operator, I want to replay recorded PV data, so that I can demonstrate the system without live sensor integration.
4. As an operator, I want replay controls, so that I can play, pause, restart, and adjust replay speed.
5. As an operator, I want to see the current anomaly score against its threshold, so that I understand why an alert is triggered.
6. As an operator, I want the GUI to show only the top-1 suggested fault, so that the interface remains simple.
7. As an operator, I want calibrated confidence shown as both a band and percentage, so that I can interpret the reliability of a suggested fault.
8. As an operator, I want severity shown as Low, Medium, High, or Critical, so that I can prioritize maintenance.
9. As an operator, I want severity to combine anomaly strength, diagnosis confidence, and persistence, so that alerts reflect operational risk.
10. As an operator, I want a persistent alert card when an anomaly occurs, so that I do not miss important events during replay.
11. As an operator, I want replay to continue when an alert occurs, so that monitoring is not interrupted.
12. As an operator, I want to acknowledge alerts, so that the system records that I have seen them.
13. As an operator, I want to add quick tags and optional notes when acknowledging alerts, so that maintenance context is preserved.
14. As an operator, I want alerts to remain active after acknowledgement until the anomaly clears, so that acknowledging does not hide unresolved problems.
15. As an operator, I want incidents to resolve automatically after the anomaly clears for a cooldown period, so that stale alerts do not remain active forever.
16. As an operator, I want to manually resolve incidents with a required note, so that I can close known events when appropriate.
17. As an admin, I want to manually resolve incidents with an optional note, so that I can override incident state when needed.
18. As an operator, I want incidents to be grouped when alerts are close in time and have the exact same fault type, so that the history is not spammed by repeated detections.
19. As an operator, I want different fault types to create separate incidents, so that distinct events are not merged incorrectly.
20. As an operator, I want low-confidence anomaly events to appear as `Unknown anomaly`, so that uncertain diagnosis is not overclaimed.
21. As an operator, I want `Unknown anomaly` incidents to remain unknown forever, so that incident records are not rewritten after the fact.
22. As an operator, I want Alert History filters by status, severity, fault type, and date/time range, so that I can find relevant incidents quickly.
23. As an operator, I want incident statuses Active, Acknowledged, and Resolved, so that the incident lifecycle is simple.
24. As an operator, I want resolved incidents never to reopen, so that each recurrence becomes a new auditable incident.
25. As an operator, I want a PDF report export, so that I can produce maintenance documentation.
26. As an operator, I want PDF reports to be maintenance-focused, so that they are useful for field inspection.
27. As an admin or thesis evaluator, I want the PDF to include a technical appendix, so that model/version traceability is preserved.
28. As an operator, I want reports generated manually only, so that the system does not create unnecessary files.
29. As an admin, I want to configure the local reports folder, so that reports are stored where deployment expects.
30. As an operator, I want the Monitor screen to show two lightweight plots, so that Jetson performance remains acceptable.
31. As an operator, I want one plot for anomaly score and another for key PV signals, so that I can relate alerts to PV behavior.
32. As an operator, I want the key PV plot to show power, irradiance, and imbalance, so that the plot is fault-oriented and understandable.
33. As an operator, I want the default plot window to show the last five minutes, so that I see recent context without overloading the Jetson.
34. As an admin, I want to configure the visible plot window duration, so that deployments can adjust for different sampling rates.
35. As an operator, I want a simple health indicator, so that I know whether the dashboard/backend/input is OK, warning, or offline.
36. As an admin, I want detailed edge performance metrics in Admin Diagnostics, so that I can validate Jetson runtime performance.
37. As an admin, I want inference latency and processing rate visible in diagnostics, so that I can confirm edge feasibility.
38. As an admin, I want Admin Mode protected by PIN/password, so that operators cannot accidentally change deployment settings.
39. As an admin, I want to manage model artifacts, so that updated Task A/Task B models can be deployed.
40. As an admin, I want to import model packages from a local folder, so that new artifacts can be loaded onto the Jetson.
41. As an admin, I want imported model packages copied into managed app storage, so that the active package does not depend on an external folder.
42. As an admin, I want imported packages activated immediately after validation, so that deployment updates are fast.
43. As an admin, I want monitoring stopped before artifact import, so that models are not swapped during active inference.
44. As an admin, I want app restart required after artifact import, so that new artifacts load safely on startup.
45. As an admin, I want schema and load validation before activation, so that invalid model packages are rejected.
46. As an admin, I want no rollback support in v1, so that artifact management remains simple.
47. As an admin, I want strict validation because rollback is absent, so that the system avoids replacing active artifacts with invalid packages.
48. As an admin, I want the app to auto-start backend and browser kiosk on boot, so that the system behaves like an edge appliance.
49. As an admin, I want backend and browser auto-restart, so that the system recovers from crashes.
50. As an admin, I want restart app and shutdown Jetson controls, so that maintenance can be performed from the GUI.
51. As an admin, I want shutdown/reboot to require PIN re-entry, so that destructive operations are protected.
52. As an admin, I want model/version metadata visible only in Admin Mode, so that operator screens remain clean.
53. As a thesis evaluator, I want model/version metadata in the PDF technical appendix, so that exported reports are traceable.
54. As a developer, I want the GUI to be local Jetson only, so that v1 avoids LAN/cloud security scope.
55. As a developer, I want SQLite persistence for incident history and settings, so that the prototype is reliable and queryable on-device.
56. As a developer, I want only operational model results stored, not raw windows, so that storage remains compact.
57. As a developer, I want full replay/raw data kept as external files referenced by path, so that SQLite remains lightweight.
58. As a developer, I want the backend separated from the frontend, so that inference, incident management, artifact validation, and UI can evolve independently.
59. As a developer, I want replay-first input abstraction, so that live input can be added later without redesigning the UI.
60. As a developer, I want no on-device training, so that Jetson Nano remains an inference/deployment target only.
61. As an admin, I want the system to monitor data/model drift, so that seasonal changes, weather shifts, sensor aging, or deployment drift can be detected.
62. As an admin, I want drift alerts to be separate from fault/anomaly alerts, so that operating-condition drift does not get confused with a PV fault incident.
63. As an admin, I want drift detection to support lightweight online detectors such as ADWIN or Page-Hinkley, so that the mechanism can run on Jetson Nano.
64. As an admin, I want drift detection to monitor model result streams and selected context streams, so that both score drift and seasonal/context drift can be detected.
65. As an admin, I want drift events to be logged for future incremental learning, so that retraining/update candidates can be reviewed offline.
66. As an admin, I want drift detection settings visible in Admin Diagnostics, so that detector choice, sensitivity, monitored streams, and recent drift events are inspectable.

## Implementation Decisions

- Use a local web application architecture: lightweight Python backend plus browser frontend in kiosk mode.
- Use local-only access in v1. The backend binds to localhost and does not expose the dashboard on the LAN.
- Use Operator Mode and Admin Mode.
- Protect Admin Mode with a simple local PIN/password.
- Use replay mode as the only v1 data source.
- Provide standard replay controls: play/pause, speed selector, restart, progress bar, and timestamp.
- Use a real-time monitoring-first main screen.
- Use a two-stage alert flow: Task A anomaly detection triggers alerting; Task B provides top-1 suggested fault if available.
- Display only top-1 diagnosis in the GUI. Do not display top-3 fault alternatives in v1.
- Use calibrated confidence band plus percentage for the suggested fault.
- Use separate calibration concepts: Task B probabilities are calibrated for operator-facing confidence, while Task A anomaly scores are thresholded/scaled and not treated as probabilities.
- Use hybrid severity based on anomaly strength, diagnosis confidence, and persistence duration.
- Use persistent alert cards that remain until acknowledged/resolved.
- Continue replay when alerts occur. Do not auto-pause or auto-open details.
- Incident grouping uses exact same fault type plus closeness in time. Related fault families are not used in v1.
- Unknown anomalies are created when Task A triggers but Task B confidence is low.
- Unknown anomaly incidents remain unknown forever. Later diagnoses may be noted but do not rewrite incident type.
- Incident lifecycle is Active, Acknowledged, Resolved.
- Incidents do not reopen after resolution.
- Incident resolution is hybrid: auto-resolve when anomaly clears for cooldown period, with manual resolve allowed by operator/admin.
- Operator manual resolve requires a note. Admin note is optional.
- Alert History supports filters by status, severity, fault type, and date/time range.
- PDF reports are generated manually only.
- PDF reports are maintenance-focused with a technical appendix.
- Reports are stored in a configurable local reports folder.
- Use SQLite for local persistence.
- SQLite stores incident summaries, acknowledgement/resolution notes, report paths, admin settings, model artifact metadata, replay session metadata, compact operational model results, drift events, and drift detector settings.
- SQLite does not store raw sensor windows, feature vectors, or full replay data.
- Store operational model result per incident: Task A anomaly score, Task A threshold, score/threshold ratio, Task A status, Task B suggested fault, calibrated confidence, confidence band, severity level, severity reason, model versions, calibration version, feature profile, input mode, and replay file path.
- Store compact drift event records: detector name, monitored stream, trigger timestamp, drift severity, detector statistic/summary, model versions, feature profile, input mode, and replay file path.
- Use five primary screens: Monitor, Alert Details, Alert History, Reports, Admin.
- Include Advanced Diagnostics as an Admin sub-section, not a main operator screen.
- Monitor screen shows system status, severity, anomaly score vs threshold, suggested top-1 fault and confidence, replay controls, latest alert card, simple system health indicator, and two plots.
- Monitor plots are anomaly score timeline and combined key PV signals plot.
- Key PV signal plot defaults to power, irradiance, and imbalance.
- Monitor plot history defaults to last five minutes and is configurable in Admin.
- Admin Diagnostics shows inference latency, processing rate, backend status, model loaded status, RAM estimate, CPU/GPU utilization if available, and dropped/late frames.
- Add a drift detection mechanism for deployment monitoring and incremental-learning support.
- Drift detection is separate from Task A fault anomaly detection. Task A answers whether the current PV behavior is faulty/anomalous; drift detection answers whether the input/model-score distribution has shifted enough that model reliability may degrade.
- Candidate drift detectors are ADWIN and Page-Hinkley, preferably through a lightweight streaming implementation such as River if dependency size/performance is acceptable on Jetson Nano.
- Drift detection should initially monitor compact streams rather than full feature vectors: Task A anomaly score, score/threshold ratio, Task B calibrated confidence, prediction entropy if available, irradiance, PV temperature, and selected normalized PV power/imbalance indicators.
- Drift events should not trigger fault incidents. They should create drift events in Admin Diagnostics and logs, with operator-visible health warning only if drift persists or reaches high severity.
- Drift events should support future incremental learning by recording when model/data drift was detected, which stream triggered it, detector state summary, and replay/live source reference.
- Incremental learning is not automatic in v1. Drift detection only flags that data should be reviewed and may justify offline model recalibration/retraining or a future artifact update.
- Admin Mode supports model artifact import from any local folder.
- Model package is copied into managed app storage before activation.
- Imported package replaces active package immediately after validation.
- No rollback in v1.
- Previous active package is deleted/replaced.
- Monitoring/replay must be stopped before model package import.
- New artifacts require app restart before becoming active.
- Model package validation uses schema + load check, not dry-run inference.
- Standard model package contents: metadata, Task A model, Task A threshold, Task A scaler, Task A feature schema, Task B model, Task B calibration artifact, Task B label mapping, and Task B feature schema.
- Artifact package import supports folder only in v1, not zip.
- Admin can select any local folder as source package.
- Jetson auto-starts full kiosk on boot.
- Backend and browser auto-restart on crash.
- Admin Mode includes Restart App and Shutdown/Reboot Jetson controls.
- Restart App requires confirmation. Shutdown/Reboot requires confirmation and Admin PIN re-entry.
- No on-device training.
- No automatic incremental learning. Drift detection can inform future offline retraining/recalibration and subsequent artifact import.
- No cloud access.
- No LAN access.

## Testing Decisions

- Tests should focus on external behavior and functional contracts, not internal implementation details.
- The incident lifecycle should be tested as a deep module: alert creation, acknowledgement, cooldown resolution, manual resolution, exact-fault-type grouping, unknown anomaly behavior, and no reopen after resolved.
- Artifact package validation should be tested as a deep module: valid package accepted, missing files rejected, invalid metadata rejected, schema mismatch rejected, invalid label mapping rejected, and import blocked while monitoring is running.
- Replay controller should be tested: play/pause, speed changes, restart, progress tracking, and end-of-replay behavior.
- Report generation should be tested: PDF created manually, report path persisted, maintenance section present, technical appendix present, and operator notes included.
- Calibration/confidence display should be tested: calibrated confidence percentage maps to Low/Medium/High band, Task A score is not displayed as probability, and severity consumes normalized anomaly strength and calibrated confidence.
- Drift detection should be tested as a deep module: no drift on stable streams, drift on synthetic mean shifts, drift on seasonal/context shifts, separate drift events from fault incidents, and persistence of drift events/settings.
- SQLite persistence should be tested: incidents persist across app restart, report metadata persists, admin settings persist, and model artifact metadata persists.
- Backend API should be tested at the contract level: monitor status endpoint, replay control endpoints, incident endpoints, report export endpoint, admin artifact validation/import endpoint, and admin diagnostics endpoint.
- Frontend smoke tests should verify Monitor loads, replay controls are usable, alert card appears, acknowledgement flow works, Alert History filters work, and Admin PIN gate works.
- Jetson-specific testing should include app starts on boot, kiosk opens dashboard, backend restart recovers, browser restart recovers, memory use remains acceptable, and UI remains responsive with five-minute plots.

## Out of Scope

- Live sensor/inverter integration.
- LAN access or cloud access.
- On-device training or fine-tuning.
- Automatic incremental learning or automatic model updates from drift events.
- Model comparison dashboard.
- Top-3 fault display.
- Rollback/version history for model packages.
- Zip model package import.
- Raw sensor/window storage in SQLite.
- Automatic PDF report generation.
- Full production authentication or role-based accounts.
- External auth integration.
- Full maintenance workflow states such as Assigned/In Progress.
- Remote monitoring from another device.
- Advanced model internals on the operator screen.
- Event-adjusted anomaly metrics.
- MLflow/DagsHub UI integration in the Jetson GUI.

## Further Notes

- The GUI should be described as a production-like thesis prototype, not a full production deployment.
- Jetson Nano 4GB is constrained. The UI should avoid heavy charts, large in-memory datasets, and unnecessary background services.
- Local web UI is preferred over a native desktop GUI because it is easier to make touch-friendly, kiosk-compatible, and modular.
- The backend should avoid loading full replay datasets into the browser. It should stream or chunk replay data and keep only a rolling buffer for plots.
- The operator screen should stay simple. Model details, calibration diagnostics, and performance metrics belong in Admin Diagnostics or PDF appendix.
- The first implementation should prioritize stable replay, alert lifecycle, artifact loading, and report export over visual polish.

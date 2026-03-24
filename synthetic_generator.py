"""
data/synthetic_generator.py

Generates synthetic labelled aviation document data for fine-tuning.
In a real scenario this would be replaced with actual document ingestion.
"""

import json
import random
import os
from pathlib import Path

LABELS = [
    "maintenance_report",
    "incident_log",
    "technical_manual",
    "safety_bulletin",
    "parts_catalog",
    "inspection_checklist",
]

TEMPLATES = {
    "maintenance_report": [
        "Aircraft {tail} underwent scheduled {check_type} maintenance on {date}. "
        "Technician replaced {component} per AMM task {task_num}. "
        "Post-maintenance run-up confirmed normal operation. MEL item {mel} cleared.",
        "Unscheduled maintenance performed on {tail} due to reported {squawk}. "
        "Root cause identified as worn {component}. P/N {pn} replaced with serviceable unit. "
        "Aircraft returned to service after ops check. MTBF data logged.",
        "Line maintenance check completed. Engine oil consumption within limits at {val} qt/hr. "
        "Brake wear indicators serviceable. All deferred defects reviewed. AOG status cleared.",
    ],
    "incident_log": [
        "Flight {flight_num} reported {event} at FL{fl}. Crew executed {procedure} checklist. "
        "ATC notified. Aircraft diverted to {airport}. No injuries. Occurrence filed per ICAO Annex 13.",
        "Ground incident at gate {gate}. {vehicle} contacted {tail} {zone} during pushback. "
        "Damage assessment initiated. Maintenance hold applied. Safety investigation opened.",
        "Bird strike reported on departure from {airport}. Engine {eng} inspected per AMM. "
        "Fan blade damage found. Engine removed and sent to MRO. SDR filed.",
    ],
    "technical_manual": [
        "Chapter {ch} — {system} System Description. The {system} system consists of {desc}. "
        "Normal operating pressure: {psi} PSI. Maximum operating altitude: {alt} ft. "
        "Refer to IPC Chapter {ch} for component identification.",
        "Task {task_num}: Removal and Installation of {component}. "
        "Required tools: {tools}. Consumables: {consumables}. "
        "Estimated manhours: {mh}. Access panel {panel} must be opened before proceeding.",
        "Troubleshooting Guide — {system} FAULT. If EICAS message {msg} is displayed, "
        "perform bite test per procedure {proc}. Check {check_items}. Expected output: {expected}.",
    ],
    "safety_bulletin": [
        "SB {sb_num}: Mandatory inspection of {component} for {defect}. "
        "Affected aircraft: {tail_range}. Compliance time: {compliance}. "
        "This bulletin is classified as ALERT due to safety implications. "
        "Contact your regional support center for spares.",
        "Airworthiness Directive {ad_num} requires modification of {system}. "
        "Failure to comply may result in {risk}. Estimated downtime: {downtime} hours. "
        "Approved repair scheme attached as Appendix A.",
        "SAFETY ALERT: Reports of {anomaly} on {fleet} fleet. "
        "Operators should increase inspection frequency on {component} from {old_freq} to {new_freq}. "
        "Engineering is investigating root cause. Updates to follow.",
    ],
    "parts_catalog": [
        "P/N {pn} — {desc}. Interchangeable with {alt_pn}. "
        "Unit of measure: {uom}. Lead time: {lead} weeks. "
        "Hazmat classification: {hazmat}. Storage temperature: {temp}.",
        "IPC Figure {fig} — {system} Assembly. Ref: {ref} — {component} — Qty {qty} per aircraft. "
        "Next higher assembly: {nha}. Obsolescence status: {status}.",
        "Rotable component P/N {pn} requires overhaul at {interval} cycles or {months} months. "
        "Approved repair vendors: {vendors}. Core exchange available. "
        "Traceability documentation required per FAA 8130-3.",
    ],
    "inspection_checklist": [
        "Pre-flight Inspection — Zone {zone}. 1. Check {item1} for security and condition. "
        "2. Verify {item2} — no cracks, corrosion or loose fasteners. "
        "3. Confirm {item3} within service limits. Sign off: ____________",
        "100-Hour Inspection Item {num}: {system} System. "
        "Inspect {component} per AMM {ref}. Acceptance criteria: {criteria}. "
        "Tools required: {tools}. Pass/Fail: ___",
        "Daily Check — Aircraft {tail}. External walk-around completed. "
        "Fluid levels checked: hydraulic {hyd}%, oil {oil} qt. "
        "Tyre pressures: NLG {nlg} PSI, MLG {mlg} PSI. Discrepancies: {disc}.",
    ],
}

FILL_VALUES = {
    "tail": ["N-XRAY42", "G-BAVO", "VT-ANQ", "A6-EMD", "9V-SKA"],
    "check_type": ["A-check", "B-check", "C-check", "line", "transit"],
    "date": ["2024-03-15", "2024-06-22", "2024-09-01", "2024-11-30"],
    "component": ["elevator actuator", "fuel control unit", "APU bleed valve", "brake assembly", "IDG"],
    "task_num": ["27-10-01", "73-20-04", "36-11-02", "32-40-07", "49-10-05"],
    "mel": ["27-10-01A", "36-11-02B", "32-40-07C"],
    "squawk": ["fuel pressure low indication", "autopilot disconnect", "gear unsafe indication"],
    "pn": ["114A1234-001", "2601234-5", "B737-800-FCU-03", "4000234-7"],
    "val": ["0.3", "0.5", "0.7", "1.1"],
    "flight_num": ["QF1", "EK501", "SQ21", "BA94", "AA100"],
    "event": ["engine flame-out", "pressurisation loss", "hydraulic system failure"],
    "fl": ["350", "310", "290", "410"],
    "procedure": ["engine relight", "emergency descent", "non-normal"],
    "airport": ["YSSY", "OMDB", "WSSS", "EGLL", "KJFK"],
    "gate": ["B12", "D5", "A42", "C9"],
    "vehicle": ["baggage tug", "fuel bowser", "GPU cart"],
    "zone": ["nose section", "wing root", "aft fuselage", "engine nacelle"],
    "eng": ["1", "2", "3", "4"],
    "ch": ["21", "27", "29", "32", "36", "49"],
    "system": ["Hydraulic", "Air Conditioning", "Flight Control", "Landing Gear", "APU"],
    "psi": ["3000", "5080", "1500", "2900"],
    "alt": ["41000", "43000", "39000"],
    "tools": ["torque wrench, safety wire pliers", "borescope, multimeter", "hoist assembly"],
    "consumables": ["O-ring kit, Loctite 243", "hydraulic fluid Skydrol", "sealant PR-1422"],
    "mh": ["2.5", "4.0", "6.5", "1.5"],
    "panel": ["417AR", "210AB", "316BL"],
    "msg": ["HYD SYS 1 LO PR", "PACK 1 FAULT", "ANTI ICE L ENG"],
    "proc": ["29-31-00 FAULT", "21-51-00 BITE"],
    "check_items": ["reservoir level, pump output, system pressure", "valve position, duct leak"],
    "expected": ["3000 PSI", "within limits", "no fault codes"],
    "sb_num": ["SB-27-1234", "SB-A320-32-0987", "SB-737-73-0521"],
    "defect": ["stress corrosion", "fretting wear", "fatigue cracking"],
    "tail_range": ["MSN 1000-2500", "all B737-800 operators", "aircraft prior to mod EO-4512"],
    "compliance": ["500 FH or 90 days", "next A-check", "within 2000 cycles"],
    "ad_num": ["2024-01-05", "2024-11-22 R1", "EASA AD 2024/1234"],
    "risk": ["uncontrolled fuel leak", "loss of directional control", "crew incapacitation"],
    "downtime": ["8", "12", "16", "24"],
    "anomaly": ["uncommanded rudder deflection", "fuel quantity discrepancy", "cabin pressure creep"],
    "fleet": ["A320", "B737 MAX", "B777", "A350"],
    "old_freq": ["500 FH", "every C-check", "annual"],
    "new_freq": ["250 FH", "every A-check", "semi-annual"],
    "alt_pn": ["114A1234-003", "2601235-5 (superseded)", "use latest dash number"],
    "uom": ["EA", "KG", "LT", "FT"],
    "lead": ["4", "8", "12", "16"],
    "hazmat": ["Class 3 Flammable", "Non-hazmat", "Class 8 Corrosive"],
    "temp": ["-20°C to +60°C", "0°C to +50°C"],
    "fig": ["32-10-01", "27-30-02", "29-10-05"],
    "ref": ["REF 1", "REF 2", "REF 3"],
    "qty": ["1", "2", "4"],
    "nha": ["Wing Trailing Edge Assembly", "Fuselage Frame 47", "Engine Pylon"],
    "status": ["Active", "Superseded by P/N 114A1234-003", "Last-time buy"],
    "interval": ["6000", "10000", "3000"],
    "months": ["60", "120", "36"],
    "vendors": ["Lufthansa Technik, ST Engineering, Air France Industries"],
    "item1": ["static wicks", "pitot covers removed", "engine inlet for FOD"],
    "item2": ["leading edge slats", "wing tip fairings", "belly antenna fairings"],
    "item3": ["tyre tread depth", "fluid levels", "brake wear indicators"],
    "num": ["1", "5", "12", "27", "38"],
    "criteria": ["no cracks >0.5 inch", "within wear limits per CMM", "no free-play"],
    "hyd": ["75", "80", "90", "100"],
    "oil": ["14", "16", "18", "20"],
    "nlg": ["180", "185", "190"],
    "mlg": ["200", "205", "210"],
    "disc": ["NIL", "see tech log ref 4521", "deferred per MEL 32-10-01"],
    "nha_field": ["Wing Assembly", "Fuselage Frame 47"],
    "desc": ["hydraulic actuators, control valves, and reservoir assembly",
             "dual-redundant pressure sensors and associated wiring harness",
             "composite structure with bonded honeycomb core"],
    "vehicle2": ["GPU", "water service truck", "lavatory service truck"],
}


def fill_template(template: str) -> str:
    """Fill template placeholders with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_dataset(n_samples: int = 800, output_path: str = "processed/dataset.jsonl"):
    """Generate a labelled dataset of aviation documents."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    samples = []

    samples_per_class = n_samples // len(LABELS)

    for label in LABELS:
        templates = TEMPLATES[label]
        for _ in range(samples_per_class):
            template = random.choice(templates)
            text = fill_template(template)
            samples.append({"text": text, "label": label})

    random.shuffle(samples)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(samples)} samples → {output_path}")
    label_counts = {}
    for s in samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    return samples


if __name__ == "__main__":
    generate_dataset(n_samples=900)

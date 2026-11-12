# SOP-MNT-102: Solar PV Inverter Fault Response and Preventive Maintenance
Revision 2 | Owner: Solar Operations | Applies to: string and central inverters

## Purpose and Scope
This procedure describes how to respond to inverter faults and how to carry out preventive maintenance at solar plants. Isolation must follow SOP-SAF-001 before any cabinet is opened.

## Common Fault Codes
E-101 indicates grid overvoltage. Check the grid voltage at the point of connection and wait for it to return to normal before restarting. E-204 indicates low insulation resistance, which means a possible ground fault on a DC string. E-310 indicates IGBT over-temperature. Check cooling fans and clean air filters.

## Response to Fault E-204
1. Isolate the inverter on both AC and DC sides and apply LOTO.
2. Disconnect each DC string one by one.
3. Measure the insulation resistance of each string with a 1000 V megger.
4. A string reading below 1 megohm is faulty. Mark it and raise a work order for the cable or module.
5. Restart the inverter only after all strings pass the test.

## Preventive Maintenance Schedule
Clean or replace inverter air filters every quarter. Perform thermal imaging of DC connectors every 6 months. Flag any connector that is more than 20 degrees C hotter than the adjacent connectors. Terminal torque must follow the OEM datasheet. This procedure does not specify torque values.

## Module Cleaning
Clean the modules when the soiling loss exceeds 4 percent. Use dry brushing or demineralised water only. Do not clean modules during the hottest part of the day, to avoid thermal shock.

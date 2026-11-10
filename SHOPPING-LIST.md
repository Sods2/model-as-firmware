# Model-as-Firmware: Shopping List

*Nov 3, 2026 · also kept as a [Claude Doc]((private link removed))*

Everything needed for Phase 0-2 on the Pico 2, all plug-in with no soldering. Prices checked 3 Nov 2026; they change, so confirm at checkout.

## What to order

Suggested split: Raspberry Pi parts from The Pi Hut (one £3.80 Royal Mail 1st Class delivery), jumper wires and power meter from Amazon UK.

| # | Item | Qty | Where to buy | What to select | Price seen | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Raspberry Pi Debug Probe (pick one shop) | 1 | [The Pi Hut](https://thepihut.com/collections/raspberry-pi/products/raspberry-pi-debug-probe) or [Amazon UK](https://www.amazon.co.uk/Raspberry-Pi-Debug-Probe/dp/B0C36HXMCB) | Nothing to choose; includes case and cables | £11.50 Pi Hut / £21.99 Amazon | To order |
| 2 | Raspberry Pi Pico 2 with headers | 3 | [The Pi Hut](https://thepihut.com/collections/raspberry-pi-store/products/raspberry-pi-pico-2) or [Amazon UK](https://www.amazon.co.uk/Raspberry-Pi-Pico-2-Headers/dp/B0FC2PQB11) | Pi Hut: choose **With Headers** | Amazon £5.91 each; Pi Hut header price not confirmed | To order |
| 3 | USB-A to Micro-USB data cable | 3 | [The Pi Hut](https://thepihut.com/collections/maker-store/products/usb-to-micro-usb-cable-0-5m) | Choose **1m** length | From £1.50 (15cm) | To order |
| 4 | Full-size breadboard (830 points) | 1 | [The Pi Hut](https://thepihut.com/collections/all/products/full-sized-breadboard) | White | £5 | To order |
| 5 | ELEGOO 120pcs jumper wires (EL-CP-004) | 1 | [Amazon UK search](https://www.amazon.co.uk/s?k=ELEGOO+120pcs+Dupont+Wire+EL-CP-004) | Male-male, male-female, female-female kit | Not checked (Pi Hut equivalent £6, sold out) | To order |
| 6 | Inline USB-A power meter | 1 | [Amazon UK search](https://www.amazon.co.uk/s?k=USB+tester+voltage+current+mWh+USB-A) | Must show **mWh** (energy) | Not checked | To order |
| 7 | Official Pi 4 power supply, UK plug (only if you lack one) | 1 | [The Pi Hut](https://thepihut.com/collections/raspberry-pi-power-supplies/products/raspberry-pi-psu-uk) | 15W USB-C, black or white | £7.50 | To order |
| 8 | SanDisk 32 GB microSD, A1 (only if you lack one) | 1 | [The Pi Hut](https://thepihut.com/products/sandisk-microsd-card-class-10-a1) | 32GB | £9 | To order |
| 9 | **Later, not now (Phase 2):** Waveshare ESP32-S3-DEV-KIT-N16R8, a second chip to check the compiler's gains carry over | 1 | [Amazon UK](https://www.amazon.co.uk/ESP32-S3-DEV-KIT-N16R8-Development-Pre-soldered-Dual-Core-Compatible/dp/B0DM4SYY36) | The listing titled "with Pre-soldered Header" (N16R8 = 16 MB flash, 8 MB PSRAM) | Not checked | Not needed |
| 10 | Later, not now (idea G): Pimoroni Pico Plus 2, adds 8 MB PSRAM and 16 MB flash | 1 | [The Pi Hut](https://thepihut.com/collections/latest-raspberry-pi-products/products/pimoroni-pico-plus-2) | Ships without headers; use it over USB-C and the debug cable only, no soldering needed | £12.90 | Not needed yet |

Estimated total for items 1-6: about £42-48 with this split, about £48-55 all on Amazon (estimates; cable and Pico prices partly assumed).

## Before you check out

- [ ] Pico 2 says **With Headers** (the cheapest Pi Hut option is without headers and needs soldering)
- [ ] Cables set to **1m**, not the default 15cm
- [ ] Not buying the separate "Male Header Set" (that is the solder-it-yourself kit)
- [ ] Power meter listing shows mWh, not only volts and amps
- [ ] If the Pi Hut "With Headers" Pico price is well above £5.91, buy the Picos on Amazon instead

**Delivery:** The Pi Hut charges £3.80 for Royal Mail 1st Class (1-2 days) or £9.99 for DHL next working day ([delivery page](https://thepihut.com/pages/delivery)). Amazon delivery depends on Prime or basket size.

**Everything here is plug-in:** the pre-soldered Picos push into the breadboard, the Debug Probe comes with JST and pin cables, and jumper wires connect the Pi 4 GPIO pins to a Pico without solder.

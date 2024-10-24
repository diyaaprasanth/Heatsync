import asyncio
from bleak import BleakClient, BleakScanner

# SwitchBot Curtain 3 service and characteristic UUIDs
SERVICE_UUID = "cba20d00-224d-11e6-9fb8-0002a5d5c51b"
CHARACTERISTIC_UUID = "cba20002-224d-11e6-9fb8-0002a5d5c51b"

# Commands
OPEN_COMMAND = bytearray([0x57, 0x0F, 0x45, 0x01, 0x05, 0xFF, 0x00])
CLOSE_COMMAND = bytearray([0x57, 0x0F, 0x45, 0x01, 0x05, 0x00, 0xFF])
STOP_COMMAND = bytearray([0x57, 0x0F, 0x45, 0x01, 0x00, 0x00, 0x00])

async def connect_and_control(device_address, command):
    device = await BleakScanner.find_device_by_address(device_address)
    if not device:
        print(f"Device with address {device_address} not found")
        return

    async with BleakClient(device) as client:
        print(f"Connected to {device.name}")

        # Send command
        await client.write_gatt_char(CHARACTERISTIC_UUID, command)
        print("Command sent successfully")

async def main():
    device_address = "AC58000B-10EB-E0D6-F033-4B2E0CF11CBE"  # Replace with your SwitchBot Curtain 3 MAC address
    
    while True:
        print("\nChoose an action:")
        print("1. Open curtain")
        print("2. Close curtain")
        print("3. Stop curtain")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        import time
        time.sleep(5)
        
        if choice == "1":
            await connect_and_control(device_address, OPEN_COMMAND)
        elif choice == "2":
            await connect_and_control(device_address, CLOSE_COMMAND)
        elif choice == "3":
            await connect_and_control(device_address, STOP_COMMAND)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())

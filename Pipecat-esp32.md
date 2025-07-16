# Pipecat ESP32 client for AI voice communication

- Run server on desktop with pipecat server.
- Run client on esp32 using pipecat-esp32 client.
- Select from multiple set of examples for different providers from examples in pipecat repo.

## Setup
- Clone pipecat repo : https://github.com/pipecat-ai/pipecat
  - `git clone --recursive https://github.com/pipecat-ai/pipecat`
- Setup esp on system following instructions : https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/linux-macos-setup.html
  - Will setup esp on base/root environment, its okay as esptool must be accessible from everywhere, conda env not required.
- Clone pipecat esp32 fork : https://github.com/harshaampar/pipecat-esp32.git
  - `git clone --recursive https://github.com/harshaampar/pipecat-esp32.git`
  - Codebug fixed and pushed in the fork
- Create a new conda env for easier python management of pipecat examples
  - `brew install --cask miniconda`
  - `conda create -n pipecat python==3.10`
  - `conda activate pipecat` : Needs to be called every time a new shell is opened.
- Install basic requirements for pipecat
  - `cd pipecat && pip install -r dev-requirements.txt`

## Basic example

### Pipecat 
Tried example : _examples/foundational/07-interruptible.py_ in pipecat repo
- Install requirements for foundational examples: `cd pipecat/examples/foundational && pip install -r requirements.txt`
- Must be inside the conda env : `conda activate pipecat`
- Uses DeepGram, Cartesia and OpenAI for STT, TTS, LLM.
- Needs all three API keys to be functional.
- Better to not hardcode APIs in repo, setup as environment variables.
  - `export DEEPGRAM_API_KEY=PASTE_API_KEY_HERE`
  - `export OPENAI_API_KEY=PASTE_API_KEY_HERE`
  - `export DEEPGRAM_API_KEY=PASTE_API_KEY_HERE`
  - If this is tough to do everytime, then create a _setup_api_key.sh_, **most importantly add it to the repo's .gitignore** before commiting, in the repo with the following code:
    - `export DEEPGRAM_API_KEY=PASTE_API_KEY_HERE && export OPENAI_API_KEY=PASTE_API_KEY_HERE && export DEEPGRAM_API_KEY=PASTE_API_KEY_HERE`
    - Then run: `source setup_api_key.sh`.
    - To make sure its been set you can run the command : `echo OPENAI_API_KEY; echo DEEPGRAM_API_KEY; echo CARTESIA_API_KEY`. It should print those key values.
  - Get the ip of machine in current network:
    - `ifconfig | grep 192.`, this will print a line with ip as 192.168.x.y, note this.
  - Run the example :
    - `python 07-interruptible.py --host 192.168.x.y --port ANY_PORT --esp32`
    - `--esp32` required for esp32, without that flag, it can be run and accessed on a local website.

### ESP32
Currently running on _esp32-s3-box_, code for the same has been pushed in the fork of the repo.
- Make sure required esp tools are setup as mentioned earlier.
- `cd pipecat-esp32/esp32-s3-box`
- Must not be in the conda environment, in the terminal it must show `(base)` before any line, if it shows `(pipecat)` then run `conda deactivate`.
- Setup variables, need to run this everyime a new shell is opened:
  - `source ../../esp/esp-idf/export.sh && export WIFI_SSID="SSID_NAME" && export WIFI_PASSWORD="SSID_PASS" && export PIPECAT_SMALLWEBRTC_URL="http://192.168.x.y/api/offer"`
  - The system where server is running and the esp32 must be in the same network(connected to the same router)
- Set target board.
  - `idf.py set-target esp32s3`
- Build the code : will take some time, around 3-4 mins.
  - `idf.py build`
- Flash to esp32 :
  - Connect esp32 to system using USB-C
  - First run `idf.py flash`
  - Sometimes on mac is necessary to specify the exact port, the output of the previous command will be :
    `Executing action: flash
    Serial port /dev/cu.usbmodem2401`, copy the port id, in this case it is `/dev/cu.usbmodem2401`, let's call it DEV_PORT_ID
  - `idf.py -p DEV_PORT_ID flash monitor`
  - This will flash and start showing the logs 
  - Press **Ctrl + ]** to close the monitor.
- esp32 will print : **ESP32 client initialized** on its screen and introduce itself over voice if everything is correctly running.
- If any of the WIFI_SSID/WIFI_PASSWORD/PIPECAT_SMALLWEBRTC_URL needs to be changed, run the previous command to set it, then:
  - `rm -rf build/ && idf.py set-target esp32s3 && idf.py build`
  - Have to rebuild the whole code, havent found a better way yet.
  - Then flash just like earlier.
- The code that is flashed will be default run on the esp32 as soon as the esp32 is powered. In case you need to access logs just run `idf.py monitor` when connected to esp32.

### Common issues
- **esp32 keeps restarting**, one of the following issues
  - Not connecting to wifi, because of wrong credentials
    - esp32 does not connect to 5GHz, needs to be 2.4GHz.
  - Not able to connect to the host because
    - The server example code(_07-interruptible.py_) is not running
    - The example code is running with a wrong IP.
    - The esp32 and the server are not in the same network.
  - Exact point of failure will be printed in the `idf.py monitor` logs.

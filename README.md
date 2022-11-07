# Safi Patel - RoboMasters Midterm 2 Practical

## Prompt 1

- For the detection prompt, I decided to base the code off what we have right now and decided to use the YOLO model we have running and whose weights are given through the link in the Google Form.
- I decided to use the YOLO model because for the simplicity it provides for my usage, it performs very well.
- In order to determine whether the plates were blue or not, I used a similar method as used in our current pipeline. However, instead of checking the total red/blue contour area in the entire frame like we currently do, I only check it within the bounding box of the detection.
    - This makes more sense since there could be a lot of other noise within the frame that doesn't have a bearing on the color of that specifically detected plate.

### Screenshot for Prompt 1:
![Prompt1](/images/prompt1.jpg)


## Prompt 2

- For the angle offset prompt, I had to just build off my prompt 1 program. In this one, I ripped out everything having to do with the depth camera since that wasn't relevant to this prompt.
- Then, I used the same way of getting the middle of the plate and calculated the angle offset using the resolution and horizontal/vertical FOVs of the camera. I realized that since I'm doing this from a sample video there isn't a way for me to get its resolution. Also the FOV is messed up since I'm doing it through my webcam.
- The calculations should be correct, however.
- I finally just displayed those angle offsets on the image itself.
    - The first number is the horizontal offset in degrees and the second number is the vertical offset.

### Screenshot for Prompt 2:
![Prompt2](/images/prompt2.jpg)

## Prompt 3
- For the SystemD prompt, I first created the startup scripts. These can be found in the `scripts` folder within this repository, but they were placed in the home directory on my computer.
    - The first startup script was `startup.sh`. This is what actually gets called by SystemD service. We need this in order to activate the conda environment before actually running the python script.
    - the second startup script was `startup.py`. This is the python file the script file ends up calling to print the PyTorch version number.
- Then, I had to make the service file which called the `startup.sh` file.
- We can see all the services listed, including mine, after I had created the service file:
![Unit List](/images/unit-list.png)
- Then I ran the commands to enable the service: 
![Unit List Enabled](/images/unit-list-enabled.png)
- After, I ran the command to start the service, I could see that it succeeded through either looking at the service status or through journalctl:
![Success](/images/system-success.png)
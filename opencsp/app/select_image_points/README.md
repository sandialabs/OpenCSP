# Open CSP directory app/image_point_select

Select points on an image by hand and save results to text file.
- 'escape' key closes the window and discards the results.
- 's' key saves selected points as a text file in the root directory.

Notes:
  1. When the window launches, it first raises a dialog to select an image file.  The dialog box only 
     shows .RAW or .NEF files.  Switch to "All files" to select images of other types (.jpg, .png, etc). 

  2. After selecting an image file, the program diplays the image.  But for some reason it shows up 
     behind all other windows.  If you minimize all other windows, then you can see the image.  
     Note that the display does not include the windows border, or minimize/maximize/close buttons, etc.

  3. There are no prompts, but the program silently waits for you to click on the image somewhere.  
     When you do, the image will be replaced by a small tile from the image in the vicinity of where 
     you clicked, shown highly magnified.  This enables you to select individual pixels, while seeing 
     surrounding context.

  4. After you make the fine-grain selction on the enlarged window, the progrem redisplays the full 
     image and waits.

  5. You can then repeat the process to select additional points.  The program is logging your selections 
     in the background.

  6. When you are done selecting points, press "s" to save and exit.  There is no confirmation raised 
     or written to the console.

  7. The program writes the selected points in a file "points_<image_name>.txt", which is written to 
     the directory from which you launched the program.  The file contains the path to the selected image.

  8. Thus, to control the location where the list of points is written, cd to the directory you wish to 
     save to, and then launch the program.  You can use the file selection dialog to navigate to the 
     image you want to select.

     For example:
        (env_310_OpenCSP) PS C:\> cd C:\ctemp\select_image_points_test\
        (env_310_OpenCSP) PS C:\ctemp\select_image_points_test> python C:\<path_to_code>\Code\OpenCSP\opencsp\app\select_image_points\SelectImagePoints.py

     The above will work if you have your default python run environment set to the virtual environment 
     env_310_OpenCSP.  If not, then you should explicitly specify this to ensure that all of the OpenCSP 
     packages are avaialble.  You can do this by:
        PS C:\ctemp\select_image_points_test> C:\<path_to_code>\Code\env_310_OpenCSP\Scripts\python.exe C:\<path_to_code>\Code\OpenCSP\opencsp\app\select_image_points\SelectImagePoints.py

     If you are unsure whether the virtual environment will be run by default, you can use the get-command 
     function:
        (env_310_OpenCSP) PS C:\ctemp\select_image_points_test> get-command python

     This will show which python executable will be run.  If the virtual environment is set as default, 
     you should see something like this:
        CommandType     Name                                               Version    Source
        -----------     ----                                               -------    ------
        Application     python.exe                                         3.10.91... C:\<path_to_code>\Code\env_310_OpenCSP\Scripts\python.exe

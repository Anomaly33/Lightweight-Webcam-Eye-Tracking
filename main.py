import os
import sys
import csv
import random
import pygame
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
from datetime import datetime
from collections import deque
from scipy.stats import beta
from pygame.locals import *
from Target import Target, TargetTest
from Gaze import Detector
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from utils import (
    get_config,
    bgr_to_rgb,
    clamp_value,
    plot_region_map,
    get_calibration_zones,
    get_undersampled_region,
)

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")
print('starting 123')
# Setup directories
data_dirs = (
    "data/l_eye",
    "data/r_eye",
    "data/face",
    "data/face_aligned",
    "data/head_pos",
)
'''
def train_models(data_file,model_type):
    models={'sgd' : SGDRegressor,
            'ridge' : Ridge,
            'mlp' : MLPRegressor,
            'svm' : SVR,
            'xgb' : XGBRegressor,
            'sgd2' : SGDRegressor,
            'ridge2' : Ridge,
            'mlp2' : MLPRegressor,
            'svm2' : SVR,
            'xgb2' : XGBRegressor}
    #print('hey')
    data = pd.read_csv(os.path.join('data_csvs',f'data_{data_file}.csv'),header=None)
    open(os.path.join('data_csvs',f'data_{data_file}.csv'),'a') 
    x=data[2868]
    y=data[2867]
    features=data[[i for i in range(0,1434)]]
    features2=data[[i for i in range(0,2868)]]
    x=x.to_numpy()
    y=y.to_numpy()
    features=features.to_numpy()
    clf_x=models[model_type]()

    clf_x.fit(features,x)
    
    clf_y=models[model_type]()
    clf_y.fit(features,y)
    
    return clf_x,clf_y
'''
def train_models(data_file, model_type):
    models = {
        'sgd': SGDRegressor,
        'ridge': Ridge,
        'mlp': MLPRegressor,
        'svm': SVR,
        'xgb': XGBRegressor,
        'sgd2': SGDRegressor,
        'ridge2': Ridge,
        'mlp2': MLPRegressor,
        'svm2': SVR,
        'xgb2': XGBRegressor,
    }

    path = os.path.join('data_csvs', f'data_{data_file}.csv')
    data = pd.read_csv(path, header=None)
    open(path, 'a')  # keep your original behavior

    n_cols = data.shape[1]
    if n_cols < 3:
        raise ValueError(f"Unexpected column count ({n_cols}). "
                         "Expected at least features + 2 targets.")

    # Last two columns are [tx, ty] because save_data does points.extend([targetx, targety])
    x = data[n_cols - 2]   # target x
    y = data[n_cols - 1]   # target y

    # All preceding columns are features
    features = data.iloc[:, : n_cols - 2]

    clf_x = models[model_type]()
    clf_x.fit(features.to_numpy(), x.to_numpy())

    clf_y = models[model_type]()
    clf_y.fit(features.to_numpy(), y.to_numpy())

    return clf_x, clf_y

def next_manual():
    if not done_dirs:
        done_dirs.append(dirt[0])
        nx,ny=manual_dirs[dirt[0]][1]
    else:
        if manual_dirs[dirt[0]][0] in done_dirs:
            collect_done[0]=True
        dirt[0]=manual_dirs[dirt[0]][0]
        nx,ny=manual_dirs[dirt[0]][1]
        done_dirs.append(dirt[0])
    return int(nx),int(ny)
'''
def save_data(data_file,points,targetx,targety):
    if points==[]:
        return num_images
    points.extend([targetx,targety])
    with open(os.path.join('data_csvs',f'data_{data_file}.csv'),'a') as f:
        writer = csv.writer(f)
        writer.writerow(points)
    region_map[
        int(targetx / SETTINGS["map_scale"]), int(targety / SETTINGS["map_scale"])
    ] += 1
    return num_images+1
'''
def save_data(data_file, points, targetx, targety):
    # no points? don't save or touch the map
    if not points:
        return num_images

    # Clamp to on-screen pixels first
    tx = max(0, min(int(targetx), w - 1))
    ty = max(0, min(int(targety), h - 1))

    # Save row
    row = list(points)
    row.extend([tx, ty])
    with open(os.path.join('data_csvs', f'data_{data_file}.csv'), 'a') as f:
        csv.writer(f).writerow(row)

    # Convert to region_map indices (and clamp to grid)
    ix = max(0, min(int(tx / SETTINGS["map_scale"]), region_map.shape[0] - 1))
    iy = max(0, min(int(ty / SETTINGS["map_scale"]), region_map.shape[1] - 1))
    region_map[ix, iy] += 1

    return num_images + 1

def cleanup():
    
    detector.close()
    pygame.quit()
    sys.exit(0)

model_x=None
model_y=None
for d in data_dirs:
    if not os.path.exists(d):
        os.makedirs(d)


# Setup pygame
pygame.init()
pygame.mouse.set_visible(0)
font_normal = pygame.font.SysFont(None, 30)
font_small = pygame.font.SysFont(None, 20)
pygame.display.set_caption("Calibrate and Collect")
#screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
w, h = pygame.display.get_surface().get_size()
center = (w // 2, h // 2)
webcam_surface = pygame.Surface(
    (SETTINGS["image_size"] * 2, SETTINGS["image_size"] * 2)
)
calibration_zones = get_calibration_zones(w, h, SETTINGS["target_radius"])

# Create target, detector, predictor
target = Target(
    center, speed=SETTINGS["target_speed"], radius=SETTINGS["target_radius"]
)

target_test = TargetTest(
    center, speed=SETTINGS["target_speed"], radius=SETTINGS["target_radius"]
)
detector = Detector(output_size=SETTINGS["image_size"])

try:
    region_map = np.load("data/region_map.npy")
except FileNotFoundError:
    region_map = np.zeros(
        (int(w / SETTINGS["map_scale"]), int(h / SETTINGS["map_scale"]))
    )

# Initialize flags and values
bg = random.choice((COLOURS["black"], COLOURS["gray"]))
bg_should_increase = True
clock = pygame.time.Clock()
ticks = 0
frame_count = 0
show_stats = False
selection_screen = True
calibrate_screen = False
calibrate_idx = 0
collect_state = 0
test_state = 0
collect_manual = [False]
collect_done = [False]
points_to_collect = SETTINGS['points_to_collect']
collect_start_region=get_undersampled_region(region_map,SETTINGS['map_scale'])
track_screen = False
test_screen = False
type_selected = False
manual_dirs = {'left':['midleft',(50,50)],
               'midleft':['down',(w/2,50)],
               'down':['middown',(w-50,50)],
               'middown':['right',(w-50,h/2)],
               'right':['midup',(w-50,h-50)],
               'midup':['up',(w/2,h-50)],
               'up':['midddl',(50,h-50)],
               'midddl':['ddl',(50,h/2)],
               'ddl':['mid',(50,50)],
               'mid':['right',(w/2,h/2)],
               }

num_images=0
done_dirs=[]
dirt=['left']
now=0
shown=0
success=0
collected_points=0
models_trained=False
done_ts = None

track_x = deque(
    [0] * SETTINGS["avg_window_length"], maxlen=SETTINGS["avg_window_length"]
)
track_y = deque(
    [0] * SETTINGS["avg_window_length"], maxlen=SETTINGS["avg_window_length"]
)
track_error = deque(
    [0] * (SETTINGS["avg_window_length"] * 2), maxlen=SETTINGS["avg_window_length"] * 2
)


while True:
    screen.fill(bg)
    presence=True
    # Vary bg colour so we get variation in ocular reflection value
    bg_origin = screen.get_at((0, 0))
    if bg_origin[0] <= COLOURS["black"][0]:
        bg_should_increase = True
    elif bg_origin[0] >= COLOURS["gray"][0]:
        bg_should_increase = False

    if bg_should_increase:
        bg = (bg_origin[0] + 1, bg_origin[1] + 1, bg_origin[2] + 1, bg_origin[3])
    else:
        bg = (bg_origin[0] - 1, bg_origin[1] - 1, bg_origin[2] - 1, bg_origin[3])

    # Get current frame from the detector
    frame_count += 1
    points = detector.get_frame()
    if points==-1:
        
        print(points)
    #else:
     #   print(detector.boxcx,detector.frame_w,detector.boxcy,detector.frame_h)
    if show_stats:
        

        num_images_text = font_normal.render(
            "# of images: {}".format(num_images),
            True,
            COLOURS["white"],
        )

        
        coverage_text = font_normal.render(
            "Coverage: {}%".format(
                round(np.count_nonzero(region_map > 0) / region_map.size * 100, 2)
            ),
            True,
            COLOURS["white"],
        )

        text_height = coverage_text.get_height()
        screen.blit(coverage_text, (10, h - text_height * 4))
        screen.blit(num_images_text, (10, h - text_height * 3))
    if selection_screen:
        text1 = font_normal.render(
            "(1) Calibrate | (2) Test | (3) Track", True, COLOURS["white"]
        )
        
        text2 = font_normal.render(
            "(s) Show stats | (esc) Quit", True, COLOURS["white"]
        )
        screen.blit(text1, (10, h * 0.3))
        screen.blit(text2, (10, h * 0.6))

        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, SETTINGS["target_radius"]
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_1:
                selection_screen = False
                calibrate_screen = True
                target.moving = False
                target.color = COLOURS["blue"]
            
            elif event.type == KEYDOWN and event.key == K_2:
                selection_screen = False
                test_screen = True
                target.color = COLOURS["red"]
            
            elif event.type == KEYDOWN and event.key == K_3:
                selection_screen = False
                track_screen = True
                target.color = COLOURS["red"]


    # Data calibration screen
    if calibrate_screen:
        target.radius=50
        if collect_done[0]:
            collect_state+=1
            collect_done[0]=False
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, SETTINGS["target_radius"]
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_SPACE:
                if collect_state==0:
                    file_name=len(os.listdir('data_csvs'))
                collect_state += 1
                target.moving = False
                collect_manual=0
            elif event.type == KEYDOWN and event.key == K_m:
                if collect_state==0:
                    file_name=len(os.listdir('data_csvs'))
                collect_state += 1
                target.moving = False
                collect_manual=1
            elif event.type == KEYDOWN and event.key == K_p:
                if collect_state==0:
                    file_name=len(os.listdir('data_csvs'))
                collect_state += 1
                target.moving = False
                collect_manual=2
                target.color=COLOURS["green"]

            elif event.type == KEYDOWN and event.key == K_r:
                if collect_state==0:
                    file_name=len(os.listdir('data_csvs'))
                collect_state +=1
                target.moving = False
                collect_manual=3
                target.color=COLOURS["green"]

        if collect_state == 0:
            
            target.x = collect_start_region[0]
            target.y = collect_start_region[1]
            target.render(screen)
            text = font_normal.render(
                "(space) random moving dot | (m) edge moving dot | (p) area points | (r) random points | (esc) Quit", True, COLOURS["white"]
            )
            screen.blit(text, (10, h * 0.3))
        elif collect_state == 1:
        
            if not target.moving and presence:
                if collect_manual==0:
                    new_x,new_y=get_undersampled_region(region_map,SETTINGS['map_scale'])
                    center = (new_x, new_y)
                elif collect_manual==1:
                    new_x,new_y=next_manual()
                    center = (new_x, new_y)
                else:
                    if int(datetime.now().timestamp()) - now>3:
                        now=int(datetime.now().timestamp())
                        if collect_manual==2:
                            new_x,new_y=next_manual()
                        else:
                            new_x,new_y=random.randint(0,w-1),random.randint(0,h-1)
                            collected_points+=1
                            if collected_points>points_to_collect:
                                collect_done[0]=True
                        center = (new_x, new_y)



            if not points:
                text = font_normal.render("Please reposition yourself", True, COLOURS["white"])
                screen.blit(text, text.get_rect(center=screen.get_rect().center))
                presence = False
            else:
                if collect_manual==0:
                    num_images = save_data(
                        file_name,
                        points,
                        target.x,
                        target.y,
                    )
                else :
                    if (int(datetime.now().timestamp()) - now)>=1:
                        num_images = save_data(
                        file_name,
                        points,
                        target.x,
                        target.y,
                    )
            if presence :
                if collect_manual<2 :
                    target.move(center, ticks)
                    
                else :
                    target.x=center[0]
                    target.y=center[1]
                target.render(screen)
                
        elif collect_state == 2:
            if done_ts is None:
                done_ts = pygame.time.get_ticks()  # start the 3s timer
                models_trained = True
                now = 0
                collected_points = 0
                done_dirs = []
                dirt = ['left']

            screen.fill(COLOURS["black"])
            text = font_normal.render("Done", True, COLOURS["white"])
            screen.blit(text, text.get_rect(center=screen.get_rect().center))

            # 3000 ms = 3 seconds
            if pygame.time.get_ticks() - done_ts >= 3000:
                # reset + go back to the main menu
                selection_screen = True
                calibrate_screen = False
                collect_state = 0
                done_ts = None

    # Track screen
    if track_screen:
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, SETTINGS["target_radius"]
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_c:
                show_webcam = not show_webcam
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_SPACE:
                selection_screen = True
                track_screen = False
                now = 0
                type_selected = False
            elif event.type == KEYDOWN and event.key == K_1:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'sgd')
            
            elif event.type == KEYDOWN and event.key == K_1:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'sgd')

            elif event.type == KEYDOWN and event.key == K_2:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'ridge')

            elif event.type == KEYDOWN and event.key == K_3:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'mlp')

            elif event.type == KEYDOWN and event.key == K_4:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'svm')

            elif event.type == KEYDOWN and event.key == K_5:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'xgb')


        if not models_trained:
            text = font_normal.render("First Calibrate please", True, COLOURS["white"])
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
                
        
        elif not type_selected:
            text = font_normal.render(" (1) SGDregressor | (2) RidgeRegressor | (3) MLPRegressor | (4) SVMRegressor", True, COLOURS["white"])
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        else:

            if not points:
                continue
            
            x_hat=model_x.predict(np.array([points]))[0]
            y_hat=model_y.predict(np.array([points]))[0]
            
            track_x.append(x_hat)
            track_y.append(y_hat)
            
            x_hat_clamp = clamp_value(x_hat, w)
            y_hat_clamp = clamp_value(y_hat, h)
            
            weights = np.arange(1, SETTINGS["avg_window_length"] + 1)
            weights_error = np.arange(1, (SETTINGS["avg_window_length"] * 2) + 1)
            #target.x = clamp_value(np.average(track_x, weights=weights),w)
            #target.y = clamp_value(np.average(track_y, weights=weights),h)
            target.x = np.average(track_x, weights=weights)
            target.y = np.average(track_y, weights=weights)
            
            target.radius = 20
            
            target.render(screen)

    if test_screen:
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, SETTINGS["target_radius"]
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_e:
                success +=1
                now=0
            elif event.type == KEYDOWN and event.key == K_SPACE:
                test_state +=1
            elif event.type == KEYDOWN and event.key == K_1:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'sgd')
                    model = 'sgd'
            elif event.type == KEYDOWN and event.key == K_2:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'ridge')
                    model = 'ridge'
            elif event.type == KEYDOWN and event.key == K_3:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'mlp')
                    model = 'mlp'
            elif event.type == KEYDOWN and event.key == K_4:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'svm')
                    model = 'svm'
            elif event.type == KEYDOWN and event.key == K_5:
                if models_trained:
                    type_selected = True
                    model_x,model_y=train_models(file_name,'xgb')

        if not models_trained:
            text = font_normal.render("First Calibrate please", True, COLOURS["white"])
            screen.blit(text, text.get_rect(center=screen.get_rect().center))

        elif not type_selected:
            text = font_normal.render(" (1) SGDregressor | (2) RidgeRegressor | (3) MLPRegressor | (4) SVMRegressor", True, COLOURS["white"])
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        
        if test_state==0 and type_selected:
             text = font_normal.render(f" Press (e) if you can successfully move the dot to the rectangle. Press SPACE to start.", True, COLOURS["white"])
             screen.blit(text, text.get_rect(center=screen.get_rect().center))
        elif test_state==1 and type_selected:
      
            if (int(datetime.now().timestamp()) - now)>3:
                now=int(datetime.now().timestamp())
                new_x,new_y=random.randint(0,w-200),random.randint(0,h-100)
                center=(new_x, new_y)
                shown+=1
            target_test.x=center[0]
            target_test.y=center[1]
            target_test.render(screen)
            target_test.color = COLOURS['green']
            
            if not points:
                continue
            
            x_hat=model_x.predict(np.array([points]))[0]
            y_hat=model_y.predict(np.array([points]))[0]
        
            track_x.append(x_hat)
            track_y.append(y_hat)
            
            x_hat_clamp = clamp_value(x_hat, w)
            y_hat_clamp = clamp_value(y_hat, h)
        
            weights = np.arange(1, SETTINGS["avg_window_length"] + 1)
            weights_error = np.arange(1, (SETTINGS["avg_window_length"] * 2) + 1)
            #target.x = clamp_value(np.average(track_x, weights=weights),w)
            #target.y = clamp_value(np.average(track_y, weights=weights),h)
            target.x = np.average(track_x, weights=weights)
            target.y = np.average(track_y, weights=weights)
            
            target.radius = 20
            
            target.render(screen)
        elif test_state==2 and type_selected :
             if shown>=1:
                 text = font_normal.render(f" ACCURACY: {success/(shown-1)}   points_tested: {shown-1}. Press SPACE to return to Menu.", True, COLOURS["white"])
                 screen.blit(text, text.get_rect(center=screen.get_rect().center))
             else :
                test_state+=1
        elif test_state>2 or (test_state>0 and not type_selected):
            selection_screen = True
            test_screen = False
            now = 0
            type_selected = False
            test_state= 0
            shown = 0
            success = 0
            
        if shown>SETTINGS['number_of_test_points'] and test_state<2 :
            test_state+=1
            with open(os.path.join('test_results',f'{model}_results.csv'),'a') as f:
                print(f'data_{file_name},{success},{shown-1}',end='\n',file=f)
              
    ticks = clock.tick(SETTINGS["record_frame_rate"])
    pygame.display.update()

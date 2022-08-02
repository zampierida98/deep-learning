"""
EfficientPose modificato per migliorare l'inferenza su immagini che non rispettano le linee guida dell'algoritmo
> python track.py --framework=pytorch --model=RT --path=utils/test1/test1_Trim.mp4

TODO: analizzare tutte le persone e non solo la più probabile (o magari solo quelle con confidenza > 0.5)
"""

# Load detection model
# https://github.com/ultralytics/yolov5/issues/2414#issuecomment-794423851
import torch
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Avoid module conflicts
# https://github.com/pytorch/hub/issues/243#issuecomment-942403391
import sys
sys.modules.pop('utils')


from getopt import getopt, error
from os.path import join, normpath
from pymediainfo import MediaInfo
import numpy as np
import time

# Import './utils/helpers.py' file
#from utils import helpers  # funziona solo nel WLS
# https://stackoverflow.com/questions/10675054/how-to-import-a-module-in-python-with-importlib-import-module
import importlib
helpers = importlib.import_module('utils.helpers')  # va aggiunto un file '__init__.py' vuoto dentro 'utils'

def get_model(framework, model_variant):
    """
    Load the desired EfficientPose model variant using the requested deep learning framework.
    
    Args:
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        model_variant: string
            EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
            
    Returns:
        Initialized EfficientPose model and corresponding resolution.
    """
    
    # Keras
    if framework in ['keras', 'k']:
        from tensorflow.keras.backend import set_learning_phase
        from tensorflow.keras.models import load_model
        set_learning_phase(0)
        model = load_model(join('models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())), custom_objects={'BilinearWeights': helpers.keras_BilinearWeights, 'Swish': helpers.Swish(helpers.eswish), 'eswish': helpers.eswish, 'swish1': helpers.swish1})
    
    # TensorFlow
    elif framework in ['tensorflow', 'tf']:
        from tensorflow.python.platform.gfile import FastGFile
        from tensorflow.compat.v1 import GraphDef
        from tensorflow.compat.v1.keras.backend import get_session
        from tensorflow import import_graph_def
        f = FastGFile(join('models', 'tensorflow', 'EfficientPose{0}.pb'.format(model_variant.upper())), 'rb')
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
        model = get_session()
        model.graph.as_default()
        import_graph_def(graph_def)
    
    # TensorFlow Lite
    elif framework in ['tensorflowlite', 'tflite']:
        from tensorflow import lite
        model = lite.Interpreter(model_path=join('models', 'tflite', 'EfficientPose{0}.tflite'.format(model_variant.upper())))
        model.allocate_tensors()
    
    # PyTorch
    elif framework in ['pytorch', 'torch']:
        from imp import load_source
        from torch import load, quantization, backends
        try:
            MainModel = load_source('MainModel', join('models', 'pytorch', 'EfficientPose{0}.py'.format(model_variant.upper())))
        except:
            print('\n##########################################################################################################')
            print('Desired model "EfficientPose{0}Lite" not available in PyTorch. Please select among "RT", "I", "II", "III" or "IV".'.format(model_variant.split('lite')[0].upper()))
            print('##########################################################################################################\n')
            return False, False
        model = load(join('models', 'pytorch', 'EfficientPose{0}'.format(model_variant.upper())))
        model.eval()
        qconfig = quantization.get_default_qconfig('qnnpack')

        # Eccezione per Windows
        import platform
        if not platform.system() == 'Windows':
            backends.quantized.engine = 'qnnpack'

    return model, {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[model_variant]

def infer(batch, model, lite, framework):
    """
    Perform inference on supplied image batch.
    
    Args:
        batch: ndarray
            Stack of preprocessed images
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        lite: boolean
            Defines if EfficientPose Lite model is used
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        
    Returns:
        EfficientPose model outputs for the supplied batch.
    """
    
    # Keras
    if framework in ['keras', 'k']:
        if lite:
            batch_outputs = model.predict(batch)
        else:
            batch_outputs = model.predict(batch)[-1]
    
    # TensorFlow
    elif framework in ['tensorflow', 'tf']:
        output_tensor = model.graph.get_tensor_by_name('upscaled_confs/BiasAdd:0')
        if lite:
            batch_outputs = model.run(output_tensor, {'input_1_0:0': batch})            
        else:
            batch_outputs = model.run(output_tensor, {'input_res1:0': batch})
    
    # TensorFlow Lite
    elif framework in ['tensorflowlite', 'tflite']:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], batch)
        model.invoke()
        batch_outputs = model.get_tensor(output_details[-1]['index'])
    
    # PyTorch
    elif framework in ['pytorch', 'torch']:
        from torch import from_numpy, autograd
        batch = np.rollaxis(batch, 3, 1)
        batch = from_numpy(batch)
        batch = autograd.Variable(batch, requires_grad=False).float()
        batch_outputs = model(batch)
        batch_outputs = batch_outputs.detach().numpy()
        batch_outputs = np.rollaxis(batch_outputs, 1, 4)
        
    return batch_outputs

def analyze_camera(model, framework, resolution, lite):
    """
    Live prediction of pose coordinates from camera.
    
    Args:
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
            
    Returns:
        Predicted pose coordinates in all frames of camera session.
    """
    
    # Load video
    import cv2
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    coordinates = []
    print('\n##########################################################################################################')
    while(True):
        
        # Read frame
        _, frame = cap.read()
        
        # Construct batch
        batch = [frame[...,::-1]]
        
        # Preprocess batch
        batch = helpers.preprocess(batch, resolution, lite)

        # Perform inference
        batch_outputs = infer(batch, model, lite, framework)

        # Extract coordinates for frame
        frame_coordinates = helpers.extract_coordinates(batch_outputs[0,...], frame_height, frame_width, real_time=True)
        coordinates += [frame_coordinates]
        
        # Draw and display predictions
        helpers.display_camera(cv2, frame, frame_coordinates, frame_height, frame_width)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
     # Print total operation time
    print('Camera operated in {0} seconds'.format(time.time() - start_time))
    print('##########################################################################################################\n')
    
    return coordinates

def analyze_image_original(file_path, model, framework, resolution, lite):
    """
    Predict pose coordinates on supplied image.
    
    Args:
        file_path: path
            System path of image to analyze
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
            
    Returns:
        Predicted pose coordinates in the supplied image.
    """
    
    # Load image
    from PIL import Image
    start_time = time.time()
    image = np.array(Image.open(file_path))
    image_height, image_width = image.shape[:2]
    batch = np.expand_dims(image, axis=0)

    # Preprocess batch
    batch = helpers.preprocess(batch, resolution, lite)
    
    # Perform inference
    batch_outputs = infer(batch, model, lite, framework)

    # Extract coordinates
    coordinates = [helpers.extract_coordinates(batch_outputs[0,...], image_height, image_width)]
    
    # Print processing time
    print('\n##########################################################################################################')
    print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
    print('##########################################################################################################\n')
    
    return coordinates
    
def analyze_video_original(file_path, model, framework, resolution, lite):
    """
    Predict pose coordinates on supplied video.
    
    Args:
        file_path: path
            System path of video to analyze
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
            
    Returns:
        Predicted pose coordinates in all frames of the supplied video.
    """
    
    # Define batch size and number of batches in each part
    batch_size = 1 if framework in ['tensorflowlite', 'tflite'] else 49
    part_size = 490 if framework in ['tensorflowlite', 'tflite'] else 10
    
    # Load video
    from skvideo.io import vreader, ffprobe
    start_time = time.time()
    try:
        videogen = vreader(file_path)
        video_metadata = ffprobe(file_path)['video']
        num_video_frames = int(video_metadata['@nb_frames'])
        num_batches = int(np.ceil(num_video_frames / batch_size))
        frame_height, frame_width = next(vreader(file_path)).shape[:2]
    except:
        print('\n##########################################################################################################')
        print('Video "{0}" could not be loaded. Please verify that the file is working.'.format(file_path))
        print('##########################################################################################################\n')
        return False
    
    # Operate on batches
    coordinates = []
    batch_num = 1
    part_start_time = time.time()
    print('\n##########################################################################################################')
    while True:

        # Fetch batch of frames
        batch = [next(videogen, None) for _ in range(batch_size)]
        if not type(batch[0]) == np.ndarray:
            break
        elif not type(batch[-1]) == np.ndarray:
            batch = [frame if type(frame) == np.ndarray else np.zeros((frame_height, frame_width, 3)) for frame in batch]

         # Preprocess batch
        batch = helpers.preprocess(batch, resolution, lite)

        # Perform inference
        batch_outputs = infer(batch, model, lite, framework)

        # Extract coordinates for batch
        batch_coordinates = [helpers.extract_coordinates(batch_outputs[n,...], frame_height, frame_width) for n in range(batch_size)]
        coordinates += batch_coordinates

        # Print partial processing time
        if batch_num % part_size == 0:
            print('{0} of {1}: Part processed in {2} seconds | Video processed for {3} seconds'.format(int(batch_num / part_size), int(np.ceil(num_batches / part_size)), '%.3f' % (time.time() - part_start_time), '%.3f' % (time.time() - start_time)))
            part_start_time = time.time()    
        batch_num += 1
    
    # Print total processing time
    print('{0} of {0}: Video processed in {1} seconds'.format(int(np.ceil(num_batches / part_size)), '%.3f' % (time.time() - start_time)))
    print('##########################################################################################################\n')
    
    return coordinates[:num_video_frames]


"""
NUOVE FUNZIONI PER LA VISUALIZZAZIONE DEI RISULTATI

Prendono in input il punto di origine del crop per posizionare i punti ed
i segmenti in maniera corretta
"""

def display_body_parts(image, image_draw, coordinates, origin, image_height=1024, image_width=1024, marker_radius=5):   
    # Define body part colors
    body_part_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw markers
    for i, (body_part, body_part_x, body_part_y) in enumerate(coordinates):
        body_part_x = origin[0] + (body_part_x * image_width)
        body_part_y = origin[1] + (body_part_y * image_height)
        image_draw.ellipse([(body_part_x - marker_radius, body_part_y - marker_radius), (body_part_x + marker_radius, body_part_y + marker_radius)], fill=body_part_colors[i])
        
    return image

def display_segments(image, image_draw, coordinates, origin, image_height=1024, image_width=1024, segment_width=5):
    # Define segments and colors
    segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)]
    segment_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw segments
    for (body_part_a_index, body_part_b_index) in segments:
        _, body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
        body_part_a_x = origin[0] + (body_part_a_x * image_width)
        body_part_a_y = origin[1] + (body_part_a_y * image_height)
        _, body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
        body_part_b_x = origin[0] + (body_part_b_x * image_width)
        body_part_b_y = origin[1] + (body_part_b_y * image_height)
        image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_colors[body_part_b_index], width=segment_width)
    
    return image

"""
NUOVe FUNZIONE PER L'ANALISI VIDEO E FOTO

TODO: rispetto all'originale non fa uso dei batch
"""

def analyze_video(file_path, model, framework, resolution, lite):
    # Load raw video
    from skvideo.io import vreader, ffprobe, FFmpegWriter
    videogen = vreader(file_path)
    video_metadata = ffprobe(file_path)['video']
    fps = video_metadata['@r_frame_rate']
    num_video_frames = int(video_metadata['@nb_frames'])
    frame_height, frame_width = next(vreader(file_path)).shape[:2]
    frame_side = frame_width if frame_width >= frame_height else frame_height

    # Initialize annotated video
    vcodec = 'libvpx-vp9' #'libx264'
    writer = FFmpegWriter(normpath(file_path.split('.')[0] + '_tracked.mp4'), inputdict={'-r': fps}, outputdict={'-r': fps, '-bitrate': '-1', '-vcodec': vcodec, '-pix_fmt': 'yuv420p', '-lossless': '1'}) #'-lossless': '1'

    # Operate on frames
    coordinates = []
    part_start_time = time.time()
    print('\n##########################################################################################################')
    from PIL import Image, ImageDraw
    i = 0
    while True:
        try:
            frame = next(videogen)
            image = np.array(Image.fromarray(frame))  # formato numpy per yolo

            # Detection
            detections = yolo(image)
            bbox = detections.pandas().xyxy[0]
            bbox = bbox.loc[bbox['name'] == 'person']

            # TODO: ciclo for su tutte (o parte) delle bbox
            bbox = bbox.loc[bbox['confidence'] == max(bbox['confidence'])]

            # Crop
            image_obj = Image.fromarray(image)  # formato pillow per fare il crop
            cropped_image = image_obj.crop((bbox.iloc[0]['xmin'],
                                            bbox.iloc[0]['ymin'],
                                            bbox.iloc[0]['xmax'],
                                            bbox.iloc[0]['ymax']))
            image = np.array(cropped_image)  # ritorno al formato numpy
            
            # Analyze image
            image_height, image_width = image.shape[:2]
            batch = np.expand_dims(image, axis=0)

            # Preprocess batch
            batch = helpers.preprocess(batch, resolution, lite)
            
            # Perform inference
            batch_outputs = infer(batch, model, lite, framework)

            # Extract coordinates
            batch_coordinates = [helpers.extract_coordinates(batch_outputs[0,...], image_height, image_width)]
            coordinates += batch_coordinates

            # Size of the detected image
            image = Image.fromarray(image)
            image_width, image_height = image.size
            image_side = image_width if image_width >= image_height else image_height

            # Annotate the original image (frame)
            image = Image.fromarray(frame)
            image_draw = ImageDraw.Draw(image)
            image_coordinates = coordinates[i]
            image = display_body_parts(image, image_draw, image_coordinates, origin=(bbox.iloc[0]['xmin'],bbox.iloc[0]['ymin']), image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
            image = display_segments(image, image_draw, image_coordinates, origin=(bbox.iloc[0]['xmin'],bbox.iloc[0]['ymin']), image_height=image_height, image_width=image_width, segment_width=int(image_side/100))
            
            # Save annotated image
            #image.save(normpath(file_path.split('.')[0] + f'_frame{i+1}_tracked.png'))
            
            # Annotate video
            writer.writeFrame(np.array(image))
            
            # Print processing time
            print('\n##########################################################################################################')
            print('Frame {0} processed in {1} seconds'.format(i+1, '%.3f' % (time.time() - part_start_time)))
            part_start_time = time.time()
            print('##########################################################################################################\n')

            i += 1
        except:
            break
    
    writer.close()

    return coordinates[:num_video_frames]

def analyze_image(file_path, model, framework, resolution, lite):
    """
    Predict pose coordinates on supplied image.
    
    Args:
        file_path: path
            System path of image to analyze
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
            
    Returns:
        Predicted pose coordinates in the supplied image.
    """
    
    # Load image
    from PIL import Image, ImageDraw
    start_time = time.time()
    original_image = Image.open(file_path)
    image = np.array(original_image)
    
    # Detection: determino la posizione delle persone all'interno dell'immagine
    detections = yolo(image)
    bbox = detections.pandas().xyxy[0]
    bbox = bbox.loc[bbox['name'] == 'person']
    coordinates = [] # inizializzo a vuoto le coordinate delle persone

    # Annotazione sulla img originale
    image_draw = ImageDraw.Draw(original_image) # immagine originale su cui fare le modifiche

    # per ogni persona riconosciuta dal detector
    for person in range(len(bbox)):
        # Crop a partire dall'immagin originale
        cropped_image = original_image.crop((bbox.iloc[person]['xmin'],
                                        bbox.iloc[person]['ymin'],
                                        bbox.iloc[person]['xmax'],
                                        bbox.iloc[person]['ymax']))
        
        image = np.array(cropped_image)  # formato numpy dell'immagine ritagliata sulla persona

        # Analisi sulla persona ritagliata + dimensioni dell'immagine
        image_height, image_width = image.shape[:2]
        batch = np.expand_dims(image, axis=0)

        # Preprocess sulla persona ritagliata
        batch = helpers.preprocess(batch, resolution, lite)
    
        # Inference sulla persona ritagliata
        batch_outputs = infer(batch, model, lite, framework)

        # Extract coordinates sulla persona ritagliata
        batch_coordinates = [helpers.extract_coordinates(batch_outputs[0,...], image_height, image_width)]
        
        # aggiungo le coordinate della persona ritagliata a coordinates che contiene le coords di tutte le persone
        # nell'immagine
        coordinates += batch_coordinates

        # lato più lungo ottenuto come la max fra la larghezza e altezza dell'immagine ritagliata
        image_side = max(image_width, image_height)

        # le coordinate di una persona soltanto
        image_coordinates = coordinates[person]

        # vado ad annotare sull'immagine originale
        display_body_parts(None, image_draw, image_coordinates, origin=(bbox.iloc[person]['xmin'],bbox.iloc[person]['ymin']), image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
        display_segments(None, image_draw, image_coordinates, origin=(bbox.iloc[person]['xmin'],bbox.iloc[person]['ymin']), image_height=image_height, image_width=image_width, segment_width=int(image_side/100))

    # Save annotated image
    original_image.save(normpath(file_path.split('.')[0] + f'_tracked.png'))
        
    # Print processing time
    print('\n##########################################################################################################')
    print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
    print('##########################################################################################################\n')
        
    return image_coordinates


def analyze(video, file_path, model, framework, resolution, lite):
    """
    Analyzes supplied camera/video/image.
    
    Args:
        video: boolean
            Flag if video is supplied, else assumes image
        file_path: path
            System path of video/image to analyze, None for camera
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
            
    Returns: 
        Predicted pose coordinates in supplied video/image.
    """
    
    # Camera-based analysis
    if file_path is None:
        coordinates = analyze_camera(model, framework, resolution, lite)
    
    # Video analysis
    elif video:
        coordinates = analyze_video(file_path, model, framework, resolution, lite)
    
    # Image analysis
    else:
        coordinates = analyze_image(file_path, model, framework, resolution, lite)
    
    return coordinates

def annotate_image(file_path, coordinates):
    """
    Annotates supplied image from predicted coordinates.
    
    Args:
        file_path: path
            System path of image to annotate
        coordinates: list
            Predicted body part coordinates for image
    """
    
    # Load raw image
    from PIL import Image, ImageDraw
    image = Image.open(file_path)
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates[0]
    image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = helpers.display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))
    
    # Save annotated image
    image.save(normpath(file_path.split('.')[0] + '_tracked.png'))
    
def annotate_video(file_path, coordinates):
    """
    Annotates supplied video from predicted coordinates.
    
    Args:
        file_path: path
            System path of video to annotate
        coordinates: list
            Predicted body part coordinates for each frame in the video
    """
    
    # Load raw video
    from skvideo.io import vreader, ffprobe, FFmpegWriter
    videogen = vreader(file_path)
    video_metadata = ffprobe(file_path)['video']
    fps = video_metadata['@r_frame_rate']
    frame_height, frame_width = next(vreader(file_path)).shape[:2]
    frame_side = frame_width if frame_width >= frame_height else frame_height

    # Initialize annotated video
    vcodec = 'libvpx-vp9' #'libx264'
    writer = FFmpegWriter(normpath(file_path.split('.')[0] + '_tracked.mp4'), inputdict={'-r': fps}, outputdict={'-r': fps, '-bitrate': '-1', '-vcodec': vcodec, '-pix_fmt': 'yuv420p', '-lossless': '1'}) #'-lossless': '1'
 
    # Annotate video
    from PIL import Image, ImageDraw
    i = 0
    while True:
        try:
            frame = next(videogen)
            image = Image.fromarray(frame) 
            image_draw = ImageDraw.Draw(image)
            image_coordinates = coordinates[i]
            image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=frame_height, image_width=frame_width, marker_radius=int(frame_side/150))
            image = helpers.display_segments(image, image_draw, image_coordinates, image_height=frame_height, image_width=frame_width, segment_width=int(frame_side/100))
            writer.writeFrame(np.array(image))
            i += 1
        except:
            break
                
    writer.close()

def annotate(video, file_path, coordinates):
    """
    Analyzes supplied video/image from predicted coordinates.
  
    Args:
        video: boolean
            Flag if video is supplied, else assumes image
        file_path: path
            System path of video/image to annotate
        coordinates: list
            Predicted body part coordinates for video/image
    """
    
    # Annotate video predictions
    if video:
        coordinates = annotate_video(file_path, coordinates)
    
    # Annotate image predictions
    else:
        coordinates = annotate_image(file_path, coordinates)
    
def save(video, file_path, coordinates):
    """
    Saves predicted coordinates as CSV.
  
    Args:
        video: boolean
            Flag if video is supplied, else assumes image
        file_path: path
            System path of video/image to annotate
        coordinates: list
            Predicted body part coordinates for video/image
    """
        
    # Initialize CSV
    import csv
    csv_path = normpath(file_path.split('.')[0] + '_coordinates.csv') if file_path is not None else normpath('camera_coordinates.csv')
    csv_file = open(csv_path, 'w')
    headers = ['frame'] if video else []
    [headers.extend([body_part + '_x', body_part + '_y']) for body_part, _, _ in coordinates[0]]
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()  

    # Write coordinates to CSV
    for i, image_coordinates in enumerate(coordinates):
        row = {'frame': i + 1} if video else {}
        for body_part, body_part_x, body_part_y in image_coordinates:
            row[body_part + '_x'] = body_part_x
            row[body_part + '_y'] = body_part_y
        writer.writerow(row)
            
    csv_file.flush()
    csv_file.close()
    
def perform_tracking(video, file_path, model_name, framework_name, visualize, store):
    """
    Process of estimating poses from camera/video/image.
    
    Args:
        video: boolean
            Flag if camera or video is supplied, else assumes image
        file_path: path
            System path of video/image to analyze, None if camera
        model_name: string
            EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework_name: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        visualize: boolean
            Flag to create visualization of predicted poses
        store: boolean
            Flag to create CSV file with predicted coordinates
            
    Returns:
        Boolean expressing if tracking was successfully performed.
    """
    
    # VERIFY FRAMEWORK AND MODEL VARIANT
    framework = framework_name.lower()
    model_variant = model_name.lower()
    if framework not in ['keras', 'k', 'tensorflow', 'tf', 'tensorflowlite', 'tflite', 'pytorch', 'torch']:
        print('\n##########################################################################################################')
        print('Desired framework "{0}" not available. Please select among "tflite", "tensorflow", "keras" or "pytorch".'.format(framework_name))
        print('##########################################################################################################\n')
        return False
    elif model_variant not in ['efficientposert', 'rt', 'efficientposei', 'i', 'efficientposeii', 'ii', 'efficientposeiii', 'iii', 'efficientposeiv', 'iv', 'efficientposert_lite', 'rt_lite', 'efficientposei_lite', 'i_lite', 'efficientposeii_lite', 'ii_lite']:
        print('\n##########################################################################################################')
        print('Desired model "{0}" not available. Please select among "RT", "I", "II", "III", "IV", "RT_Lite", "I_Lite" or "II_Lite".'.format(model_name))
        print('##########################################################################################################\n')
        return False
        
    # LOAD MODEL
    else:
        model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant 
        lite = True if model_variant.endswith('_lite') else False
        model, resolution = get_model(framework, model_variant)
        if not model:
            return True
        
    # PERFORM INFERENCE
    coordinates = analyze(video, file_path, model, framework, resolution, lite)
        
    # VISUALIZE PREDICTIONS
    if visualize and file_path is not None and coordinates:
        annotate(video, file_path, coordinates)
        
    # STORE PREDICTIONS AS CSV
    if store and coordinates:
        save(video, file_path, coordinates)
        
    return True
        
def main(file_path, model_name, framework_name, visualize, store):
    """
    Main program for performing tracking from camera or video or pose estimation of image.
    
    Args:
        file_path: path/string
            System path of video/image to analyze, None to perform live tracking
        model_name: string
            EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        framework_name: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        visualize: boolean
            Flag to create visualization of predicted poses
        store: boolean
            Flag to create CSV file with predicted coordinates
    """
    
    # LIVE ANALYSIS FROM CAMERA
    if file_path is None:
        print('\n##########################################################################################################')
        print('Click "Q" to end camera-based tracking.'.format(file_path))
        print('##########################################################################################################\n')
        perform_tracking(video=True, file_path=None, model_name=model_name, framework_name=framework_name, visualize=False, store=store)

    # VIDEO ANALYSIS
    elif 'Video' in [track.track_type for track in MediaInfo.parse(file_path).tracks]:
        perform_tracking(video=True, file_path=normpath(file_path), model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)
        
    # IMAGE ANALYSIS
    elif 'Image' in [track.track_type for track in MediaInfo.parse(file_path).tracks]:
        perform_tracking(video=False, file_path=normpath(file_path), model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)
      
    else:
        print('\n##########################################################################################################')
        print('Ensure supplied file "{0}" is a video or image'.format(file_path))
        print('##########################################################################################################\n')
    
if __name__== '__main__':
      
    # Fetch arguments
    args = sys.argv[1:]
    
    # Define options
    short_options = 'p:m:f:vs'
    long_options = ['path=', 'model=', 'framework=', 'visualize', 'store']
    try:
        arguments, values = getopt(args, short_options, long_options)
    except error as err:
        print('\n##########################################################################################################')
        print(str(err))
        print('##########################################################################################################\n')
        sys.exit(2)
    
    # Define default choices
    file_path = None
    model_name = 'I_Lite'
    framework_name = 'TFLite'
    visualize = False
    store = False

    # Set custom choices
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--path'):
            file_path = current_value if len(current_value) > 0 else None
        elif current_argument in ('-m', '--model'):
            model_name = current_value
        elif current_argument in ('-f', '--framework'):
            framework_name = current_value
        elif current_argument in ('-v', '--visualize'):
            visualize = True
        elif current_argument in ('-s', '--store'):
            store = True
    print('\n##########################################################################################################')
    print('The program will attempt to analyze {0} using the "{1}" framework with model "{2}", and the user did{3} like to store the predictions and wanted{4} to visualize the result.'.format('"' + file_path + '"' if file_path is not None else 'the camera', framework_name, model_name, '' if store else ' not', '' if visualize or file_path is None else ' not'))
    print('##########################################################################################################\n')
        
    main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)

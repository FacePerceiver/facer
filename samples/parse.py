import facer

image = facer.hwc2bchw(facer.read('sample.jpg')) # image: 1 x 3 x h x w

face_detector = facer.face_detector('retinaface/mobilenet', device='cpu')
faces = face_detector(image) # faces: [{'rects': n x 4, 'points': n x 5 x 2}]. Note: len(faces) should always be identical to image.shape[0].

# face_recognizer = facer.face_recognizer('arcface', device='cpu')
# faces = face_recognizer(image, faces) # faces: [{'rects': n x 4, 'points': n x 5 x 2, 'id_vec': n x 256}]

# face_landmark_detector = facer.face_landmark_detector('farl/aflw19', device='cpu')
# faces = face_landmark_detector(image, faces=faces) # faces: [{'rects': n x 4, 'points': n x 5 x 2, 'id_vec': n x 256, 'landmarks': n x 19 x 2}]

# face_segmenter = facer.face_segmenter('farl/celebm', device='cpu')
# faces = face_segmenter(image, faces=faces) # faces: [{'rects': n x 4, 'points': n x 5 x 2, 'id_vec': n x 256, 'landmarks': n x 19 x 2, 'masks': n x h x w x 1}]

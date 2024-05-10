import argparse
import cv2
import numpy as np
from Object_Loader import OBJECT

# Constants
MINIMUM_MATCHES_REQUIRED = 10
DEFAULT_FILL_COLOR = (0, 0, 0)
CAMERA_MATRIX = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])


def initialize_feature_detector_and_matcher():
    """Initialize ORB feature detector and Brute Force matcher."""
    orb_detector = cv2.ORB.create()
    brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return orb_detector, brute_force_matcher


def load_reference_image(filepath):
    """Load an image from a given filepath."""
    print("Loading reference image from:", filepath)
    reference_image = cv2.imread(filepath, 0)
    if reference_image is None:
        raise FileNotFoundError(f"Unable to load the reference image at {filepath}")
    return reference_image


def load_3d_object_model(filepath, swap_yz_axes):
    """Load a 3D object model from a file."""
    return OBJECT(filepath, swap_yz_axes)


def detect_and_compute_keypoints(orb_detector, brute_force_matcher, reference_image, current_frame):
    """Detect and compute keypoints and their descriptors in images."""
    keypoints_reference, descriptors_reference = orb_detector.detectAndCompute(reference_image, None)
    keypoints_frame, descriptors_frame = orb_detector.detectAndCompute(current_frame, None)
    matches = brute_force_matcher.match(descriptors_reference, descriptors_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    return keypoints_reference, keypoints_frame, matches


def compute_homography_matrix(keypoints_reference, keypoints_frame, matches):
    """Compute the homography matrix from matched keypoints."""
    source_points = np.float32([keypoints_reference[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography_matrix, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    return homography_matrix


def draw_detected_rectangle(frame, reference_image, homography_matrix):
    """Draw a rectangle on the detected plane in the current video frame."""
    height, width = reference_image.shape
    reference_border_points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1,
                                                                                                                     1,
                                                                                                                     2)
    transformed_dst = cv2.perspectiveTransform(reference_border_points, homography_matrix)
    frame = cv2.polylines(frame, [np.int32(transformed_dst)], True, 255, 3, cv2.LINE_AA)
    return frame


def render_object_model(frame, object_model, projection_matrix, reference_image, use_color=False):
    """Render a 3D object model onto the current video frame based on the projection matrix."""
    vertices = object_model.vertices
    scale_matrix = np.eye(3) * 3
    height, width = reference_image.shape

    for face in object_model.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + width / 2, p[1] + height / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection_matrix)
        img_pts = np.int32(dst)
        if not use_color:
            cv2.fillConvexPoly(frame, img_pts, DEFAULT_FILL_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # Convert to BGR
            cv2.fillConvexPoly(frame, img_pts, color)
    return frame


def compute_projection_matrix(camera_parameters, homography_matrix):
    """Compute the projection matrix for rendering the 3D model."""
    homography_matrix = -homography_matrix
    rotation_and_translation = np.dot(np.linalg.inv(camera_parameters), homography_matrix)
    rotation_1 = rotation_and_translation[:, 0] / np.linalg.norm(rotation_and_translation[:, 0])
    rotation_2 = rotation_and_translation[:, 1] / np.linalg.norm(rotation_and_translation[:, 1])

    rotation_2 = rotation_2 - np.dot(rotation_2, rotation_1) * rotation_1
    rotation_2 = rotation_2 / np.linalg.norm(rotation_2)
    rotation_3 = np.cross(rotation_1, rotation_2)

    projection_matrix = np.stack((rotation_1, rotation_2, rotation_3, rotation_and_translation[:, 2])).T
    return np.dot(camera_parameters, projection_matrix)


def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    hex_length = len(hex_color)
    return tuple(int(hex_color[i:i + hex_length // 3], 16) for i in range(0, hex_length, hex_length // 3))


def parse_command_line_arguments():
    """Parse command line arguments for the AR application."""
    parser = argparse.ArgumentParser(description='Augmented reality application')
    parser.add_argument('-r', '--rectangle', help='Draw rectangle delimiting target surface on frame',
                        action='store_true')
    parser.add_argument('-mk', '--model_keypoints', help='Draw model keypoints', action='store_true')
    parser.add_argument('-fk', '--frame_keypoints', help='Draw frame keypoints', action='store_true')
    parser.add_argument('-ma', '--matches', help='Draw matches between keypoints', action='store_true')
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    orb_detector, brute_force_matcher = initialize_feature_detector_and_matcher()
    reference_image = load_reference_image(r'E:\Everything\Projects\nitinkumar1014\AR_Basic\Reference_Planes\Plane.jpg')
    object_model = load_3d_object_model(r'E:\Everything\Projects\nitinkumar1014\AR_Basic\Models\fox.obj',
                                        swap_yz_axes=True)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return -1

    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            print("Unable to capture video")
            return -1

        keypoints_reference, keypoints_frame, matches = detect_and_compute_keypoints(orb_detector, brute_force_matcher,
                                                                                     reference_image, current_frame)
        if len(matches) > MINIMUM_MATCHES_REQUIRED:
            homography_matrix = compute_homography_matrix(keypoints_reference, keypoints_frame, matches)
            if homography_matrix is not None:
                if args.rectangle:
                    current_frame = draw_detected_rectangle(current_frame, reference_image, homography_matrix)
                projection_matrix = compute_projection_matrix(CAMERA_MATRIX, homography_matrix)
                current_frame = render_object_model(current_frame, object_model, projection_matrix, reference_image,
                                                    use_color=False)

            if args.matches:
                current_frame = cv2.drawMatches(reference_image, keypoints_reference, current_frame, keypoints_frame,
                                                matches[:10], 0, flags=2)

            cv2.imshow('frame', current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Not enough matches found - {len(matches)}/{MINIMUM_MATCHES_REQUIRED}")

    video_capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()

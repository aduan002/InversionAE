from tqdm import tqdm
import numpy as np
import os

class AnomalyGeneration:
    def __init__(self, node_path, element_path, translation_path, output_dir) -> None:
        self.nodes = np.genfromtxt(fname = node_path,usecols=(1,2,3),skip_header=1,skip_footer=1)
        
        translations = np.genfromtxt(fname = translation_path)
        for i in range(3):
            self.nodes[:,i] = self.nodes[:, i] + translations[i] # Nodes are encoded as an offset of the translations.
        
        self.elements = np.genfromtxt(fname = element_path,dtype=np.int32,skip_header=1,usecols=(1,2,3,4))
        self.elements = self.elements - 1 # 1-indexing to 0-indexing

        self.zones = np.genfromtxt(fname = element_path,dtype=np.int32, skip_header=1, usecols=5)

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.midpoints = None

    def compute_midpoints(self):
        print("Computing midpoints:")
        NUM_VERTICES = 4 # Tetrahedron shape has 4 vertices
        midpoints = np.zeros((self.elements.shape[0], 3))
        for i in tqdm(range(self.elements.shape[0])):
            for j in range(3):
                midpoints[i,j] = (1/NUM_VERTICES) * sum(self.nodes[self.elements[i,:], j])
        self.midpoints = midpoints

    def generate_anomaly(self, file_path, leak_point = (436650,368210,88), distance = 2, factor = 1.3):
        # distance is in meters.
        # factor - 1 > 0 is a percent increase, factor - 1 < 0 is a percent decrease.
        distances_to_leak = np.sqrt(
            (self.midpoints[:, 0] - leak_point[0]) ** 2 + 
            (self.midpoints[:, 1] - leak_point[1]) ** 2 + 
            (self.midpoints[:, 2] - leak_point[2]) ** 2 
        )
        leak_index = np.argmin(distances_to_leak[np.where( (self.zones == 4) | (self.zones == 5) )])
        leak_point = self.midpoints[leak_index] # Find the closest midpoint to the original leak_point and choose that new point as our leak point.

        distances_to_leak = np.sqrt(
            (self.midpoints[:, 0] - leak_point[0]) ** 2 + 
            (self.midpoints[:, 1] - leak_point[1]) ** 2 + 
            (self.midpoints[:, 2] - leak_point[2]) ** 2 
        )

        #affected_idx = np.where((distances_to_leak <= distance) & ( (self.zones == 4) | (self.zones == 5) ))[0] # Zones 4 and 5 are below the cap.
        zone_4 = np.where(self.zones==4)
        zone_5 = np.where(self.zones==5)

        sigma_input = np.genfromtxt(fname = file_path, skip_header=0)
        #sigma_input[affected_idx] = factor * sigma_input[affected_idx]
        #sigma_input = factor * sigma_input
        #sigma_input = 0 * sigma_input
        sigma_input[zone_4] = factor * sigma_input[zone_4]
        sigma_input[zone_5] = factor * sigma_input[zone_5]
        
        file_dir, file_name = os.path.split(file_path)
        np.savetxt(os.path.join(self.output_dir, file_name), sigma_input, fmt='%.8E')

    def generate_anomalies(self, file_dir, distance = 2, factor = 1.3):
        print("Generating anomalies:")
        log_info = {}
        file_names = os.listdir(file_dir)

        indices = np.random.choice(self.nodes.shape[0], len(file_names), replace=False)
        idx = 0
        for file_name in tqdm(file_names):
            x,y,z = self.nodes[indices[idx]]
            leak_point = (x,y,z)

            self.generate_anomaly(file_path=os.path.join("sigmas", file_name), leak_point=leak_point, distance=distance, factor=factor)
            idx += 1

            log_info[file_name] = leak_point

        with open("generate_anomalies.log", "w") as file:
            for file_name in log_info:
                x,y,z = log_info[file_name]

                line = "{0} {1} {2} {3}\n".format(file_name, x,y,z)
                file.write(line)





if __name__ == "__main__":
    node_path = "F3B.1.node"
    element_path = "F3B.1.ele"
    translation_path = "F3B.trn"
    output_dir = "Anomalies_val_zones=4,5_factor=1.3" # "Anomalies"

    file_name = "FA3_8L20220928_2001.sig"
    file_dir = "filtered_data/val"
    leak_point = (436650,368210,88)
    distance = 40
    factor = 1.3

    generator = AnomalyGeneration(node_path=node_path, element_path=element_path, translation_path=translation_path, output_dir=output_dir)
    generator.compute_midpoints()
    #generator.generate_anomaly(file_name=file_name, leak_point=leak_point, distance=distance, factor=factor)
    generator.generate_anomalies(file_dir=file_dir, distance=distance, factor=factor)


#define NDEBUG
#include "PxPhysicsAPI.h"
#include <iostream>
#include <fstream>
#include "wavefront.h"
#include <Eigen/Dense>
#include <math.h>
#include <chrono>

#define CONFIG_FILE_PATH "/path/to/config"

using namespace physx;

// Initialize the required structures 

PxDefaultAllocator		gAllocator;
PxDefaultErrorCallback	gErrorCallback;
PxFoundation*           gFoundation = NULL;
PxPhysics*				gPhysics	= NULL;
PxCooking*				gCooking	= NULL;

uint number_of_meshes;
uint number_of_sensors;
std::string mesh_path;
std::string rotations_file_path;
std::string landmarks_path;

// Load the configuration file parameters
bool LoadConfigurationFile()
{
    std::ifstream cFile (CONFIG_FILE_PATH);
    if (!cFile.is_open())
    {
        return false;
    }
    std::string param;

    while(!cFile.eof())
    {
        cFile >> param;

        if (param == "NUMBER_OF_MESHES")
        {
            cFile >> number_of_meshes;
            std::cout<<number_of_meshes<<std::endl;
        }
        if (param == "NUMBER_OF_SENSORS")
        {
            cFile >> number_of_sensors;
            std::cout<<number_of_sensors<<std::endl;
        }
        if (param == "MESH_PATH")
        {
            cFile >> mesh_path;
            std::cout<<mesh_path<<std::endl;
        }
        if (param == "ROTATIONS_FILE_PATH")
        {
            cFile >> rotations_file_path;
            std::cout<<rotations_file_path<<std::endl;;
        }
        if (param == "LANDMARKS_PATH")
        {
            cFile >> landmarks_path;
            std::cout<<landmarks_path<<std::endl;;
        }
    }
    return true;

}

int main (int argc, char** argv)
{

    // Init the top level PxPhysics object

    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    if(!gFoundation)
        std::cout<<"PxCreateFoundation failed!"<<std::endl;

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(),true);


    if(!gPhysics)
		exit(0);

	// Init the cooking library
	gCooking = PxCreateCooking(PX_PHYSICS_VERSION, *gFoundation, 
                               PxCookingParams(PxTolerancesScale()));

    // Init the vector to store convex meshes
    std::vector<PxConvexMeshGeometry> convex_geom_vector(number_of_meshes);
    if (!LoadConfigurationFile())
    {
        std::cout<< "Error with configuration file"<<std::endl;
        return 1;
    }

    for (int iter = 0; iter < number_of_meshes; iter++)
    {
        // Initialize the wrapper class for the mesh
        WavefrontObj obj_iter;
        int we = obj_iter.loadObj((mesh_path + std::to_string(iter) + ".obj").c_str(), false);

        PxVec3*  vertices_iter = new PxVec3[obj_iter.mVertexCount];
        PxU32* indices_iter = new PxU32[obj_iter.mTriCount * 3];

        for (int i = 0; i < obj_iter.mVertexCount ; i++)
        {
                PxVec3 v(obj_iter.mVertices[i*3], obj_iter.mVertices[i*3+1],
                         obj_iter.mVertices[i*3+2]);
                vertices_iter[i] = v;
        }

        for (int i = 0; i < obj_iter.mTriCount; i++)
        {
            for (int j = 0; j<3; j++)
            {
                indices_iter[i*3 +j] = obj_iter.mIndices[i*3 +j];
            }
        }

        // Create the convex mesh
        PxConvexMesh* convex_iter = NULL;
        PxConvexMeshDesc desc_iter;
        desc_iter.points.data = vertices_iter;
        desc_iter.points.count = obj_iter.mVertexCount;
        desc_iter.points.stride = sizeof(PxVec3);
        desc_iter.indices.count = obj_iter.mTriCount;
        desc_iter.indices.data = indices_iter;
        desc_iter.indices.stride = 3 * sizeof(PxU32);
        desc_iter.flags = PxConvexFlag::eCOMPUTE_CONVEX;
        convex_iter = gCooking->createConvexMesh(desc_iter, gPhysics->getPhysicsInsertionCallback());
        PxConvexMeshGeometry convex_geom_iter(convex_iter);
        convex_geom_vector.push_back(convex_geom_iter);
    }

    std::cout<< "Loaded " + std::to_string(number_of_meshes) + " meshes"<<std::endl;

    // Read and store the sensors poses
    std::ifstream file_sensors (landmarks_path); 

    std::vector<std::vector<float>> vectors_sensors;
    std::string line_sensors;
    while (std::getline(file_sensors, line_sensors))
    {
        std::istringstream ss(line_sensors);
        std::vector<float> new_vec;
        float v;

        while(ss >> v)
        {
            new_vec.push_back(v);

        }
        vectors_sensors.push_back(new_vec);
    }
    file_sensors.close();

    // Create the simple box for the DIGIT
    PxBoxGeometry box_digit(0.0161, 0.0135, 0.017);

	PxTransform sensors_transformation[vectors_sensors.size()];

    for (int i = 0; i < vectors_sensors.size(); i++)
    {
        // Assign the traslation vector
        PxVec3 translation(vectors_sensors[i][0], vectors_sensors[i][1], vectors_sensors[i][2]);

        // Assign the rotation matrix
        PxQuat quaternion(vectors_sensors[i][6], PxVec3(vectors_sensors[i][3], vectors_sensors[i][4],
                          vectors_sensors[i][5]));
        PxMat33 matrix (quaternion);

        // Add the necessary translation due to the different position of the object reference 
        // system
        PxVec3 delta(0.0035, 0, 0.017);
        translation += matrix * delta;

        // Save the transformationss
        sensors_transformation[i] = PxTransform(translation, quaternion);
    }

    // Loop over the 15 different starting positions
	for (int k = 0; k< 15; k++)
	{
		// Read from rotations file
		std::ifstream file ((rotations_file_path + "rotations_best" + 
                             std::to_string(k) + ".txt").c_str());
		std::vector<std::vector<float>> vectors;
		std::string line;
        std::cout<<k<<std::endl;

        while (std::getline(file, line))
		{
			std::istringstream ss(line);
			std::vector<float> new_vec;
			float v;

			while(ss >> v)
			{
				new_vec.push_back(v);
			}
			vectors.push_back(new_vec);
		}

        file.close();

		std::vector<float> depths(vectors.size());

// Parallelize the computation for the poses
#pragma omp parallel for
		for(int i =0; i < vectors.size(); i++)
		{
            // Initialize the vectors to store the direction, the depth and the penetration boolean
            std::vector<std::vector<PxVec3>> direction_vectors;
            std::vector<std::vector<PxF32>> depth_vectors;

            for (int we = 0; we < number_of_sensors; we++)
            {
                std::vector<PxVec3> direction_vector(number_of_meshes);
                std::vector<PxF32> depth_vector(number_of_meshes);
                direction_vectors.push_back(direction_vector);
                depth_vectors.push_back(depth_vector);
             }

            std::vector<bool> isPen(number_of_meshes);

            //Assign the traslation vector
            PxVec3 translation(vectors[i][0], vectors[i][1], vectors[i][2]);

			// Assign the rotation matrix
			float norm = std::sqrt(std::pow(vectors[i][3], 2) + std::pow(vectors[i][4], 2) + 
                                   std::pow(vectors[i][5], 2));
			PxQuat quaternion(norm, PxVec3(vectors[i][3]/norm, vectors[i][4]/norm,
                              vectors[i][5]/norm));
			PxMat33 matrix (quaternion);

            // Initialize the final depth
			float final_depth = 0;

			// Save the object transform
			PxTransform temp_transform (translation, quaternion);

            for(int z = 0; z < number_of_meshes; z++)
            {
                for (int sens = 0; sens < number_of_sensors; sens++)
                {
                    // Compute the Penetration
                    isPen[z] = PxGeometryQuery::computePenetration(direction_vectors[sens][z],
                                                                   depth_vectors[sens][z],
                                                                   convex_geom_vector[z],
                                                                   temp_transform,
                                                                   box_digit,
                                                                   sensors_transformation[sens]);
                    if(isPen[z])
                    {   // Check whether this sensor has a greater penetration tha the others
                        if (final_depth < depth_vectors[sens][z])
                        {

                            final_depth = depth_vectors[sens][z];

                        }
                    }
                }
            }
            // Store the final depth
			depths.at(i) = final_depth;
		}

        // Save the file with the depths for every initial position
		std::fstream filev;
		filev.open("depths"+ std::to_string(k)+".txt",std::ios_base::out);
		for(int i=0;i<depths.size();i++)
		{
			filev<<depths[i]<<std::endl;
		}
	}

    // Free the memory
    gCooking->release();
    gPhysics->release();
    gFoundation->release();

    return 0;
}

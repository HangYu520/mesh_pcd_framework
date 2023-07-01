#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include "Type.h"
#include "TriMesh.h"
#include "PointCloud.h"

std::vector<std::string> getAllFiles(const std::string& folder_path) {
    std::vector<std::string> files;
    
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_directory()) {
            // get all files in the sub-directory
            std::vector<std::string> subFiles = getAllFiles(entry.path().string());
            files.insert(files.end(), subFiles.begin(), subFiles.end());
        }
        else if (entry.is_regular_file()) {
            //regular file
            files.push_back(entry.path().string());
        }
    }

    return files;
}

void backgound_running(const std::string& folder_path)
{
    /*
    * This function computes a 2d car point cloud dataset from a 3d car point cloud dataset.
    * All the 3d car point clouds in the folder_path are processed in background one by one.
    */

    //get all files in the folder_path
    std::vector<std::string> files = getAllFiles(folder_path);
    for (int i = 0; i < files.size(); i++)
    {
        std::string file = files[i];
        PointCloud pointcloud;

        //print info
        std::stringstream ss;
        ss << i + 1 << "/" << files.size() << " " << file << ":\n";
        std::cout << std::endl << ss.str();

        //read the file
        pointcloud.ReadFromFile(file.c_str());
        std::cout << "read point cloud successfully!\n";
        std::cout << "points size: " << pointcloud.n_points() << std::endl;

        //bounding box
        Bbox_3 bbx = pointcloud.BoundingBox();
        double lengths[3] = { bbx.xmax() - bbx.xmin(),
                              bbx.ymax() - bbx.ymin(),
                              bbx.zmax() - bbx.zmin() };
        std::unordered_map<int, char> dirs;
        dirs[0] = 'x';
        dirs[1] = 'y';
        dirs[2] = 'z';
        int max_id;
        double max_val = -1;
        for (int i = 0; i < 3; i++)
        {
            if (lengths[i] > max_val)
            {
                max_val = lengths[i];
                max_id = i;
            }
        }
        std::cout << "max dir: " << dirs[max_id] << " length: " << max_val << std::endl;

        //projection
        char dir = dirs[max_id];
        if (dir == 'x')
            pointcloud.Project_along_y();
        else if (dir == 'y')
            pointcloud.Project_along_x();
        else if (dir == 'z')
            pointcloud.Project_along_x();
        std::cout << "projection completed!\n";

        //boundary points
        double alpha = 5;
        int sampled_points = 1000;
        AlphaShapeData asd = pointcloud.alpha_shape_2d(alpha);
        std::cout << "recompute with alpha = " << asd.optimal_alpha * 10 << std::endl;
        AlphaShapeData asd2 = pointcloud.alpha_shape_2d(asd.optimal_alpha * 10);
        std::vector<Segment_2> segments = asd2.segments;
        std::vector<Segment_2> c_segments = pointcloud.connect_segments(segments);
        std::cout << "num of boundary points: " << c_segments.size() << std::endl;//print info
        std::vector<Point_2> d_points = pointcloud.downsample(c_segments, sampled_points);
        std::cout << "num of sampled boundary points: " << d_points.size() << std::endl;//print info

         //scale the points
        pointcloud.clear();
        for (Point_2 point : d_points)
        {
            pointcloud.insert(Point_3(point.x(), point.y(), 0));
        }
        double scaled_length = 200;
        double scale_ratio = scaled_length / max_val;
        pointcloud.scale(scale_ratio);
        std::cout << "scale the points with ratio = " << scale_ratio << std::endl;

        //save points to xyz file
        std::ofstream xyzfile;
        std::string saved_folder = "res/pcd/car/temp/";
        int lastBackslashPos = file.find_last_of("\\");
        std::string xyzfilepath = file.substr(lastBackslashPos + 1);
        xyzfile.open(saved_folder + xyzfilepath);
        for (int i = 0; i < pointcloud.n_points(); i++)
        {
            xyzfile << std::setprecision(12) << pointcloud.Point(i).x() << " "
                << pointcloud.Point(i).y() << " 0 255 0 0\n";
        }
        xyzfile.close();
        std::cout << "points saved to " << saved_folder + xyzfilepath << std::endl;
    }
}

int main(int argc, char *argv[])
{
    //Geometry Definition
    TriMesh mesh;
    PointCloud pointcloud;
    
	igl::opengl::glfw::Viewer viewer;
 
    //Lambda function
    auto ReadFile = ([&] {
        assert(argc > 2);
        const char* filepath = argv[2];
        if (strcmp(argv[1], "-mesh") == 0)
            mesh.ReadFromFile(filepath);
        else if (strcmp(argv[1], "-pcd") == 0)
            pointcloud.ReadFromFile(filepath);
        }); //read file via command line

    auto Update = ([&] {
        mesh.Draw(viewer);
        pointcloud.Draw(viewer);
        }); //update the viewer widget
	
    //read file
    ReadFile();
    
    //ImGui settings
	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	viewer.plugins.push_back(&plugin);
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);
    viewer.core().background_color = Eigen::Vector4f(1, 1, 1, 0); //default background color is white
    viewer.data().label_size = 2.f;
    viewer.data().line_width = 2.f;
                                                                
    // Customize the menu
    double doubleVariable = 0.1f; // Shared between two menus

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();
        ImGui::DragFloat("Label size", &viewer.data().label_size, 1.f, 0.f, 10.f);
        ImGui::DragFloat("Line width", &viewer.data().line_width, 1.f, 0.f, 10.f);
    };
    
    // Draw custom windows
    menu.callback_draw_custom_window = [&]()
    {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
        
        //Pointcloud window
        ImGui::Begin("PointCloud", nullptr, ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("PointCloud Info"))
        {
            std::cout << "points:" << pointcloud.n_points() << std::endl;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save"))
        {
            pointcloud.WriteXYZ("res/pcd/save/save.xyz");
        }
        if (ImGui::Button("AverageSpacing"))
        {
            std::cout << "average spacing: " << pointcloud.AverageSpacing() << std::endl;
        }
        ImGui::SameLine();
        if (ImGui::Button("EstimateNormal"))
        {
            pointcloud.EstimateNormal();
        }
        ImGui::DragFloat("Pointsize", &viewer.data().point_size, 0.5f, 0.f, 10.f);
        if (ImGui::Button("DrawNormal"))
        {
            pointcloud.DrawNormal(viewer);
        }
        ImGui::SameLine();
        if (ImGui::Button("clear Line"))
        {
            viewer.data().clear_edges();
            viewer.data().clear_labels();
        }
        if (ImGui::Button("PoissonRecon"))
        {
            Mesh output_mesh = pointcloud.PoissonRecon();
            mesh.SetMesh(output_mesh);
            Update();
        }
        ImGui::SameLine();
        if (ImGui::Button("IterPoissonRecon"))
        {
            auto mesh_arr = pointcloud.IterPoissonRecon();
            for (int i = 0; i < mesh_arr.size(); i++)
            {
                std::stringstream ss;
                ss << "res/mesh/save/" << i << ".obj";
                mesh.SetMesh(mesh_arr[i]);
                mesh.writeOBJ(ss.str().c_str());
            }
            Update();
        }
        if (ImGui::Button("GaussNoise"))
        {
            //set mu and sigma
            double mu, sigma;
            std::cout << "Please input mean(mu) and variance(sigma):";
            std::cin >> mu >> sigma;
            pointcloud.addGaussNoise(mu, sigma);
            Update();
        }
        ImGui::SameLine();
        if (ImGui::Button("addOutliers"))
        {
            double ratio;
            std::cout << "Please input ratio of outliers:";
            std::cin >> ratio;
            pointcloud.addOutliers(ratio);
            Update();
        }
        ImGui::SameLine();
        if (ImGui::Button("MainDir"))
        {
            pointcloud.DrawMainDir(viewer);
            Update();
        }
        if (ImGui::Button("Axis"))
        {
            pointcloud.DrawAxis(viewer);
        }
        ImGui::SameLine();
        if (ImGui::Button("Projection"))
        {
            std::cout << "choose a direction to project(x,y,z):";
            char dir;
            std::cin >> dir;
            if (dir == 'x')
                pointcloud.Project_along_x();
            else if (dir == 'y')
                pointcloud.Project_along_y();
            else if (dir == 'z')
                pointcloud.Project_along_z();
            Update();
        }
        ImGui::SameLine();
        if (ImGui::Button("boundary points"))
        {
            std::cout << "Please input the required num of boundary points: ";
            int sampled_points;
            std::cin >> sampled_points;
            std::cout << "Please input the alpha in alpha shape computation: ";
            double alpha;
            std::cin >> alpha;
            pointcloud.DrawBoundary(viewer, alpha, sampled_points);
            Update();
        }
        if (ImGui::Button("bounding box"))
        {
            pointcloud.DrawBoundingBox(viewer);
        }
        ImGui::SameLine();
        if (ImGui::Button("scale"))
        {
            std::cout << "Please input the direction: ";
            char dir;
            std::cin >> dir;
            std::cout << "Please input the scaled length: ";
            double length;
            std::cin >> length;
            double ratio;
            Bbox_3 bbx = pointcloud.BoundingBox();
            if (dir == 'x')
                ratio = length / (bbx.xmax() - bbx.xmin());
            else if (dir == 'y')
                ratio = length / (bbx.ymax() - bbx.ymin());
            else if (dir == 'z')
                ratio = length / (bbx.zmax() - bbx.zmin());
            pointcloud.scale(ratio);
            viewer.data().clear_edges();
            viewer.data().clear_labels();
            Update();
            pointcloud.DrawAxis(viewer);
            pointcloud.DrawBoundingBox(viewer);
        }
        if (ImGui::Button("remove duplicate points"))
        {
            int original_points = pointcloud.n_points();
            pointcloud.removeDuplicatePoints();
            int new_points = pointcloud.n_points();
            if (original_points == new_points)
                std::cout << "No duplicate points!\n";
            else
                std::cout << "remove " << original_points - new_points << " points\n";
            Update();
        }
        if (ImGui::Button("background running"))
        {
            std::cout << "Please input the folder path: ";
            std::string folder_path;
            std::cin >> folder_path;
            backgound_running(folder_path);
        }
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling() + 400, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 280), ImGuiCond_FirstUseEver);

        //Mesh window
        ImGui::Begin("TriMesh", nullptr, ImGuiWindowFlags_NoSavedSettings);

        // Expose the same variable directly ...
        if (ImGui::Button("Mesh Info"))
        {
            std::cout << "vertices:" << mesh.n_vertices() << std::endl;
            std::cout << "faces:" << mesh.n_faces() << std::endl;
        }
        if (ImGui::Button("Laplace Smooth"))
        {
            mesh.LaplaceSmooth();
            Update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Loop Subdivision"))
        {
            mesh.LoopSubdiv();
            Update();
        }
        if (ImGui::Button("Draw Segmentation"))
        {
            //get segmentation txt file
            std::cout << "Please input the txt file path:";
            std::string filepath, line;
            std::cin >> filepath;
            std::ifstream file(filepath);
            std::vector<int> faceseg;
            while (std::getline(file, line))
            {
                faceseg.push_back(std::stoi(line));
            }
            Update();
            mesh.DrawSegmentation(viewer, faceseg);
        }
        ImGui::SameLine();
        if (ImGui::Button("Draw Data"))
        {
            //get data file patt
            std::cout << "Please input the txt file path:";
            std::string filepath;
            std::cin >> filepath;
            std::ifstream file(filepath);
            std::vector<double> data;
            while (!file.eof())
            {
                double d;
                file >> d;
                data.push_back(d);
            }
            mesh.DrawData(viewer, data);
        }
        if (ImGui::Button("boundary"))
        {
            mesh.Drawboundary(viewer);
        }
        ImGui::End();
    };
    
    //draw geometry
    Update();
    
	viewer.launch();

}

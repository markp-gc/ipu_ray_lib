// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <scene_utils.hpp>
#include <ipu_utils.hpp>

#include <exception>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


void transform(HostTriangleMesh& mesh,
               std::function<void(embree_utils::Vec3fa&)>&& tfVerts,
               std::function<void(embree_utils::Vec3fa&)>&& tfNormals) {
  for (auto& v : mesh.vertices) {
    tfVerts(v);
  }

  for (auto& n : mesh.normals) {
    tfNormals(n);
  }

  mesh.updateBoundingBox();
  ipu_utils::logger()->debug("New bounding box for mesh: {} {} {} -> {} {} {}",
    mesh.getBoundingBox().min.x, mesh.getBoundingBox().min.y, mesh.getBoundingBox().min.z,
    mesh.getBoundingBox().max.x, mesh.getBoundingBox().max.y, mesh.getBoundingBox().max.z);
}

void addQuad(HostTriangleMesh& mesh, const std::vector<embree_utils::Vec3fa>& verts) {
  if (verts.size() != 4) {
    throw std::logic_error("Quad must have 4 vertices.");
  }

  // Triangulate the quad:
  std::uint32_t vertOffset = mesh.vertices.size();
  mesh.vertices.push_back(verts[0]);
  mesh.vertices.push_back(verts[1]);
  mesh.vertices.push_back(verts[2]);
  mesh.vertices.push_back(verts[3]);

  mesh.triangles.push_back(Triangle(0 + vertOffset, 1 + vertOffset, 2 + vertOffset));
  mesh.triangles.push_back(Triangle(2 + vertOffset, 3 + vertOffset, 0 + vertOffset));
}

HostTriangleMesh makeGroundPlane(const embree_utils::Vec3fa& pos, float scale) {
  HostTriangleMesh mesh;
  addQuad(mesh, {
    {pos.x + -scale, pos.y, pos.z + -scale},
    {pos.x + -scale, pos.y, pos.z + +scale},
    {pos.x + +scale, pos.y, pos.z + -scale},
    {pos.x + +scale, pos.y, pos.z + +scale}
  });
  return mesh;
}

// Copy mesh data from aiScene data structure into ours:
void getMeshes(const aiScene* aiFile, std::vector<HostTriangleMesh>& meshes, std::vector<std::uint32_t>& matIDs, bool loadNormals) {
  // Get meshes:
  meshes.reserve(meshes.size() + aiFile->mNumMeshes);
  matIDs.reserve(matIDs.size() + aiFile->mNumMeshes);
  std::uint32_t faceCount = 0;

  for (auto m = 0u; m < aiFile->mNumMeshes; ++m) {
    auto& mesh = *aiFile->mMeshes[m];
    auto& mat = *aiFile->mMaterials[mesh.mMaterialIndex];
    matIDs.push_back(mesh.mMaterialIndex);
    ipu_utils::logger()->debug("Mesh {} '{}' has {} faces. Material: {} '{}'", m, mesh.mName.C_Str(), mesh.mNumFaces, mesh.mMaterialIndex, mat.GetName().C_Str());
    faceCount += mesh.mNumFaces;

    meshes.push_back(HostTriangleMesh());
    auto& hostMesh = meshes.back();
    for (auto f = 0u; f < mesh.mNumFaces; ++f) {
      const auto& face = mesh.mFaces[f];
      if (face.mNumIndices != 3) {
        throw std::runtime_error("Only triangle meshes are supported.");
      }
      hostMesh.triangles.push_back(Triangle(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
    }
    for (auto v = 0u; v < mesh.mNumVertices; ++v) {
      auto& vert = mesh.mVertices[v];
      hostMesh.vertices.push_back(embree_utils::Vec3fa(vert[0], vert[1], vert[2]));
    }
    if (loadNormals && mesh.HasNormals()) {
      ipu_utils::logger()->debug("Loading normals for mesh {} '{}'", m, mesh.mName.C_Str());
      for (auto v = 0u; v < mesh.mNumVertices; ++v) {
        auto& n = mesh.mNormals[v];
        hostMesh.normals.push_back(embree_utils::Vec3fa(n[0], n[1], n[2]));
      }
    } else {
      ipu_utils::logger()->debug("No normals loaded for mesh {} '{}'", m, mesh.mName.C_Str());
    }
    hostMesh.updateBoundingBox();
    ipu_utils::logger()->debug("Bounding box for mesh {}: {} {} {} -> {} {} {}", m,
      hostMesh.getBoundingBox().min.x, hostMesh.getBoundingBox().min.y, hostMesh.getBoundingBox().min.z,
      hostMesh.getBoundingBox().max.x, hostMesh.getBoundingBox().max.y, hostMesh.getBoundingBox().max.z);
  }

  ipu_utils::logger()->info("Loaded {} faces.", faceCount);
}

void importMesh(std::string& filename, std::vector<HostTriangleMesh>& meshes) {
  // Load a mesh:
  if (!filename.empty()) {
    Assimp::Importer importer;
    const auto* aiFile = importer.ReadFile(filename,
      aiProcess_PreTransformVertices   |
      aiProcess_OptimizeMeshes         |
      aiProcess_CalcTangentSpace       |
      aiProcess_Triangulate            |
      aiProcess_JoinIdenticalVertices  |
      aiProcess_SortByPType);

    if (aiFile) {
      // Get meshes from the file:
      ipu_utils::logger()->info("Found {} meshes in file '{}'.", aiFile->mNumMeshes, filename);
      std::vector<HostTriangleMesh> importedMeshes;
      std::vector<std::uint32_t> importedMatIds;
      const bool loadNormals = false;
      getMeshes(aiFile, importedMeshes, importedMatIds, loadNormals);

      // Apply hardcoded transforms to position the monkey bust mesh and then append
      // the transformed meshes to existing meshes:
      for (auto& hostMesh : importedMeshes) {
        // Scale mesh to reasonable scale for box scene and place on short box:
        auto diag = hostMesh.getBoundingBox().max - hostMesh.getBoundingBox().min;
        auto scale = 175.f / std::sqrt(diag.squaredNorm());
        transform(hostMesh,
          [&](embree_utils::Vec3fa& v) {
            v.x = -v.x;
            v.z = -v.z; // Rotate 180 so monkey faces camera
            v *= scale; // Scale
            v += embree_utils::Vec3fa(210, 165, 160); // Translate to top of box
          },
          [&](embree_utils::Vec3fa& n) {
            // Inverse transpose rotation happens to be same:
            n.x = -n.x;
            n.z = -n.z;
          }
        );

        meshes.push_back(hostMesh);
      }

    } else  {
      throw std::runtime_error(importer.GetErrorString());
    }
  }
}

// Load a complete scene with camera attempting to reinterpret materials:
SceneDescription importScene(std::string& filename, bool loadNormals) {
  SceneDescription scene;

  Assimp::Importer importer;
  const auto* aiFile = importer.ReadFile(filename,
    aiProcess_PreTransformVertices   |
    aiProcess_OptimizeMeshes         |
    aiProcess_CalcTangentSpace       |
    aiProcess_Triangulate            |
    aiProcess_JoinIdenticalVertices  |
    aiProcess_SortByPType);

  if (aiFile == nullptr) {
    ipu_utils::logger()->error("Could not load file '{}'", filename);
    throw std::runtime_error("Could not load scene file.");
  }

  ipu_utils::logger()->info("Importing scene from file '{}'", filename);
  ipu_utils::logger()->info("Found {} meshes", aiFile->mNumMeshes);
  ipu_utils::logger()->info("Found {} cameras", aiFile->mNumCameras);
  ipu_utils::logger()->info("Found {} materials", aiFile->mNumMaterials);
  ipu_utils::logger()->info("Found {} lights (ignored)", aiFile->mNumLights);
  ipu_utils::logger()->info("Found {} textures: (ignored)", aiFile->mNumTextures);

  // Read cameras:
  if (aiFile->mNumCameras < 1) {
    ipu_utils::logger()->error("Scene must contain at least one camera");
    throw std::runtime_error("No camera found in scene file.");
  }

  for (auto c = 0u; c < aiFile->mNumCameras; ++c) {
    auto& camera = *aiFile->mCameras[c];
    const std::string camName = camera.mName.C_Str();

    ipu_utils::logger()->debug("Camera {} name: '{}'", c, camName);
    ipu_utils::logger()->debug("Camera '{}' horizontal fov (radians): {}", camName, camera.mHorizontalFOV);

    aiMatrix4x4 cm;
    camera.GetCameraMatrix(cm);
    aiVector3D p = camera.mPosition;
    aiVector3D l = camera.mLookAt;
    ipu_utils::logger()->debug("Camera '{}' position: {}, {}, {}", camName, p.x, p.y, p.z);
    ipu_utils::logger()->debug("Camera '{}' lookat: {}, {}, {}", camName, l.x, l.y, l.z);
    ipu_utils::logger()->debug("Camera '{}' matrix: {}, {}, {}, {}", camName, cm.a1, cm.a2, cm.a3, cm.a4);
    ipu_utils::logger()->debug("Camera '{}' matrix: {}, {}, {}, {}", camName, cm.b1, cm.b2, cm.b3, cm.b4);
    ipu_utils::logger()->debug("Camera '{}' matrix: {}, {}, {}, {}", camName, cm.c1, cm.c2, cm.c3, cm.c4);
    ipu_utils::logger()->debug("Camera '{}' matrix: {}, {}, {}, {}", camName, cm.d1, cm.d2, cm.d3, cm.d4);

    if (c == 0) {
      if (aiFile->mNumCameras > 1) {
        ipu_utils::logger()->warn("Scene contains more than one camera: using '{}'", camName);
      }
      scene.camera.horizontalFov = camera.mHorizontalFOV;
      scene.camera.matrix = {cm.a1, cm.a2, cm.a3, cm.a4, cm.b1, cm.b2, cm.b3, cm.b4, cm.c1, cm.c2, cm.c3, cm.c4, cm.d1, cm.d2, cm.d3, cm.d4};
    }
  }

  // Get materials:
  scene.materials.reserve(aiFile->mNumMaterials);

  for (auto m = 0u; m < aiFile->mNumMaterials; ++m) {
    auto& mat = *aiFile->mMaterials[m];
    const std::string matName = mat.GetName().C_Str();
    ipu_utils::logger()->debug("Material {} name: '{}'", m, matName);

    scene.materials.emplace_back();
    auto& newMaterial = scene.materials.back();

    aiColor3D col;
    auto err = mat.Get(AI_MATKEY_COLOR_DIFFUSE, col);
    if (err == AI_SUCCESS) {
      newMaterial.albedo = embree_utils::Vec3fa(col.r, col.g, col.b);
      ipu_utils::logger()->debug("Material '{}' diffuse: {}, {}, {}",
                                  matName, col.r, col.g, col.b);
    }

    mat.Get(AI_MATKEY_COLOR_EMISSIVE, col);
    if (err == AI_SUCCESS) {
      newMaterial.emission = embree_utils::Vec3fa(col.r, col.g, col.b);
      ipu_utils::logger()->debug("Material '{}' emission: {}, {}, {}",
                                  matName, col.r, col.g, col.b);
      if (newMaterial.emission.isNonZero()) {
        newMaterial.emissive = true;
      }
    }

    err = mat.Get(AI_MATKEY_REFRACTI, newMaterial.ior);
    if (err == AI_SUCCESS) {
      ipu_utils::logger()->debug("Material '{}' refractive index: {}", matName, newMaterial.ior);
    }

    // In absence of emission factor in importer use shininess
    // as emission factor. This needs to be faked it when the file
    // is exported or modify after (e.g. for dae files you can add
    // it to the markdown by hand):
    if (newMaterial.emissive) {
      float emissionFactor = 1.f;
      err = mat.Get(AI_MATKEY_SHININESS, emissionFactor);
      if (err == AI_SUCCESS) {
        newMaterial.emission *= emissionFactor;
        ipu_utils::logger()->debug("Material '{}' shininess: {}", matName, emissionFactor);
        ipu_utils::logger()->warn("Material '{}' shininess ({}) will be used as emission factor", matName, emissionFactor);
        ipu_utils::logger()->trace("Material '{}' emission {}, {}, {}", matName, newMaterial.emission.x, newMaterial.emission.y, newMaterial.emission.z);
      }
    }

    float transparency = 0.f;
    err = mat.Get(AI_MATKEY_TRANSPARENCYFACTOR, transparency);
    if (err == AI_SUCCESS) {
      // Assume transparent materials refract.
      newMaterial.type = Material::Type::Refractive;
      ipu_utils::logger()->debug("Material '{}' transparency: {}", matName, transparency);
      ipu_utils::logger()->debug("Material '{}' interpretted as DIELECTRIC", matName);
    }
    // Transparency is not parsed from collada so add a hack to read glass material from scene:
    if (matName.find("glass") != std::string::npos) {
      newMaterial.type = Material::Type::Refractive;
      ipu_utils::logger()->debug("Material '{}' interpretted as DIELECTRIC", matName);
    }

    float reflectivity = 0.f;
    err = mat.Get(AI_MATKEY_REFLECTIVITY, reflectivity);
    if (err == AI_SUCCESS) {
      if (reflectivity > 0.f) {
        // Any reflectivity impplies material is specular:
        newMaterial.type = Material::Type::Specular;
        ipu_utils::logger()->debug("Material '{}' interpretted as SPECULAR", matName);
      }
      ipu_utils::logger()->debug("Material '{}' reflectivity: {}", matName, reflectivity);
    }
  }

  getMeshes(aiFile, scene.meshes, scene.matIDs, loadNormals);

  // Transform scene so camera is at origin looking straight down -z axis:
  aiMatrix4x4t<float> cm(
    scene.camera.matrix[0], scene.camera.matrix[1], scene.camera.matrix[2], scene.camera.matrix[3],
    scene.camera.matrix[4], scene.camera.matrix[5], scene.camera.matrix[6], scene.camera.matrix[7],
    scene.camera.matrix[8], scene.camera.matrix[9], scene.camera.matrix[10], scene.camera.matrix[11],
    scene.camera.matrix[12], scene.camera.matrix[13], scene.camera.matrix[14], scene.camera.matrix[15]);

  // Extract the inverse transpose rotation to transform mesh normals:
  aiMatrix4x4t<float> cmInvT = cm;
  aiVector3t<float> scaling;
  aiQuaterniont<float> rotation;
  aiVector3t<float> translation;
  cmInvT.Inverse();
  cmInvT.Transpose();
  cmInvT.Decompose(scaling, rotation, translation);

  auto tfv = [&](embree_utils::Vec3fa& v) {
    auto p = cm * aiVector3D(v.x, v.y, v.z);
    v = embree_utils::Vec3fa(-p.x, p.y, -p.z); // Swap handedness
  };

  auto tfn = [&](embree_utils::Vec3fa& n) {
    auto p = rotation.Rotate(aiVector3D(n.x, n.y, n.z));
    n = embree_utils::Vec3fa(-p.x, p.y, -p.z); // Swap handedness
  };

  for (auto& m : scene.meshes) {
    transform(m, tfv, tfn);
  }

  return scene;
}

std::vector<HostTriangleMesh> makeCornellBox() {
  HostTriangleMesh light;
  HostTriangleMesh white;
  HostTriangleMesh red;
  HostTriangleMesh green;

  // Light:
  addQuad(light, {
    {343, 548.7998, 227},
    {343, 548.7998, 332},
    {213, 548.7998, 332},
    {213, 548.7998, 227}
  });

  // Floor:
  addQuad(white, {
    {552.8, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 559.2},
    {549.6, 0.0, 559.2}
  });

  // Ceiling:
  addQuad(white, {
    {556, 548.8, 0},
    {556, 548.8, 559.2},
    {0, 548.8, 559.2},
    {0, 548.8, 0}
  });

  // Back wall:
  addQuad(white, {
    {549.6, 0, 559.2},
    {0, 0, 559.2},
    {0, 548.8, 559.2},
    {556, 548.8, 559.2}
  });

  // Right wall:
  addQuad(green, {
    {0, 0, 559.2},
    {0, 0, 0},
    {0, 548.8, 0},
    {0, 548.8, 559.2}
  });

  // Left wall:
  addQuad(red, {
    {552.8, 0, 0},
    {549.6, 0, 559.2},
    {556, 548.8, 559.2},
    {556, 548.8, 0}
  });

  return std::vector<HostTriangleMesh>{light, white, red, green};
}

HostTriangleMesh makeCornellShortBlock() {
  HostTriangleMesh block;

  addQuad(block, {
    {130, 165, 65},
    {82, 165, 225},
    {240, 165, 272},
    {290, 165, 114}
  });

  addQuad(block, {
    {290, 0, 114},
    {290, 165, 114},
    {240, 165, 272},
    {240, 0, 272}
  });

  addQuad(block, {
    {130, 0, 65},
    {130, 165, 65},
    {290, 165, 114},
    {290, 0, 114}
  });

  addQuad(block, {
    {82, 0, 225},
    {82, 165, 225},
    {130, 165, 65},
    {130, 0, 65}
  });

  addQuad(block, {
    {240, 0, 272},
    {240, 165, 272},
    {82, 165, 225},
    {82, 0, 225}
  });

  return block;
}

HostTriangleMesh makeCornellTallBlock() {
  HostTriangleMesh block;

  addQuad(block, {
    {423, 330, 247},
    {265, 330, 296},
    {314, 330, 456},
    {472, 330, 406}
  });

  addQuad(block, {
    {423, 0, 247},
    {423, 330, 247},
    {472, 330, 406},
    {472, 0, 406}
  });

  addQuad(block, {
    {472, 0, 406},
    {472, 330, 406},
    {314, 330, 456},
    {314, 0, 456}
  });

  addQuad(block, {
    {314, 0, 456},
    {314, 330, 456},
    {265, 330, 296},
    {265, 0, 296}
  });

  addQuad(block, {
    {265, 0, 296},
    {265, 330, 296},
    {423, 330, 247},
    {423, 0, 247}
  });

  return block;
}

SceneDescription makeCornellBoxScene(std::string& meshFile, bool boxOnly) {
  SceneDescription scene;

  scene.meshes = makeCornellBox();
  scene.meshes.push_back(makeCornellShortBlock());
  scene.meshes.push_back(makeCornellTallBlock());

  if (!boxOnly) {
    // Add a few other primitives:
    scene.spheres.emplace_back(embree_utils::Vec3fa(450.f, 37.f, 90.f), 37.f);
    scene.spheres.emplace_back(embree_utils::Vec3fa(350.f, 37.f, 90.f), 37.f);
    scene.discs.emplace_back(embree_utils::Vec3fa(1, 0, 0), embree_utils::Vec3fa(0.0002f, 300.f, 250.f), 60.f);
    importMesh(meshFile, scene.meshes);
  }

  ipu_utils::logger()->debug("Number of triangle meshes in scene: {}", scene.meshes.size());

  // Transform scene so camera is at origin and change handedness of coordinate system:
  embree_utils::Vec3fa cameraPosition(278, 273, -800); // From Cornell spec.
  auto tfv = [&](embree_utils::Vec3fa& v) {
    v -= cameraPosition;
    v.x = -v.x;
    v.z = -v.z;
  };

  auto tfn = [&](embree_utils::Vec3fa& n) {
    // Normals are disabled for built int scene
  };

  for (auto& m : scene.meshes) {
    transform(m, tfv, tfn);
  }

  for (auto& s : scene.spheres) {
    s.x -= cameraPosition.x;
    s.y -= cameraPosition.y;
    s.z -= cameraPosition.z;
    s.x = -s.x;
    s.z = -s.z;
  }

  for (auto& d : scene.discs) {
    d.cx -= cameraPosition.x;
    d.cy -= cameraPosition.y;
    d.cz -= cameraPosition.z;
    d.cx = -d.cx;
    d.cz = -d.cz;
    d.nx = -d.nx;
    d.nz = -d.nz;
  }

  // Define materials:
  const auto black  = embree_utils::Vec3fa(0.f, 0.f, 0.f);
  const auto red   = embree_utils::Vec3fa(.66f, 0.f, 0.f);
  const auto green = embree_utils::Vec3fa(0.f, .48f, 0.f);
  const auto blue  = embree_utils::Vec3fa(0.4f, 0.4f, .85f);
  const auto blueLight  = embree_utils::Vec3fa(0.4f, 0.7f, .92f) * 2.f;
  const auto white = embree_utils::Vec3fa(.75f, .75f, .75f);
  const auto grey = embree_utils::Vec3fa(.4f, .4f, .4f);
  const auto lightR = embree_utils::Vec3fa(0.78f, 0.78f, 0.78f);
  const auto lightE = embree_utils::Vec3fa(
    (100.f * 15.6f + 100.f * 18.4f) / 255.f,
    (100.f * 8.f + 74.5f * 15.6f) / 255.f,
    (57.3f * 8.f) / 255.f
  );

  scene.materials = std::vector<Material>{
    Material(white, black, Material::Type::Diffuse),
    Material(red, black, Material::Type::Diffuse),
    Material(green, black, Material::Type::Diffuse),
    Material(blue, black, Material::Type::Refractive),
    Material(lightR, lightE, Material::Type::Diffuse),
    Material(grey, black, Material::Type::Specular),
    Material(blue, blueLight, Material::Type::Diffuse),
    Material(blue, black, Material::Type::Diffuse)
  };

  // Assign materials to primitives:
  // Note badness: if you load a file with more than two meshes you need to add more material indices here
  scene.matIDs = {
    // light, white-box-parts, left-wall, right-wall, short-box, tall-box:
    4, 0, 1, 2, 0, 5,
    // Loaded Meshes (unfortunately we hard code materials for loaded meshes):
    0, 0,
    // Sphere, sphere, disc
    3, 7, 6
  };

  const auto numPrims = scene.meshes.size() + scene.spheres.size() + scene.discs.size();
  if (scene.matIDs.size() < numPrims) {
    throw std::logic_error("All primitives must be assigned a material.");
  }

  scene.camera.horizontalFov = embree_utils::Piby4;

  return scene;
}


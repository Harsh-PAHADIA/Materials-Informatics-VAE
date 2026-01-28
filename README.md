# Materials-Informatics-VAE

# 🔬 Generative Materials Informatics: Synthesis & Property Prediction

### **Project Overview**
This project demonstrates a full-stack **Materials Informatics** pipeline. It utilizes a **Variational Autoencoder (VAE)** to learn representations of metallic microstructures from the **NIST UHCS dataset**. 

---

### Key Technical Features**
* **Generative AI:** Convolutional VAE in **PyTorch** mapping SEM micrographs to a 16-dimensional latent space.
* **Data Scale:** Trained on **76,000+ microstructure patches** for high-fidelity synthesis.
* **Property Prediction:** Real-time correlation of latent features with mechanical properties:
    * **Yield Strength ($\sigma_y$):** Modeled via Hall-Petch relationship.
    * **Vickers Hardness (HV):** Derived from phase morphology.
    * **Ductility (%):** Capturing the classic strength-ductility trade-off.

---

### The Physics: Hall-Petch Logic**
The prediction engine simulates the Hall-Petch effect, where decreasing grain size increases strength:
$$\sigma_y = \sigma_0 + k_y d^{-1/2}$$

---

### Developer: Harshita Pahadia**
* **IIT Jodhpur:** B.Tech in Materials and Metallurgical Engineering (Minor in AI)
* **Focus:** Computational Metallurgy & Generative AI

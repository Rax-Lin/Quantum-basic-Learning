## L1: Complex Numbers (複數基礎)

### 核心觀念
- **簡介**：量子態的 Probability Amplitude 是以複數表示。量子閘的運算（如 Hadamard, Phase Shift）本質上就是複數矩陣與複數向量的乘法。
- **幾何意義與單位圓（Unit Circle）**：
  - 複數可以透過歐拉公式轉換為極座標形式 $c = r \cdot e^{i\theta}$。
  - 在量子系統中，為了維持機率守恆，複數向量通常在**單位圓 ($r=1$)** 上進行操作，量子閘的影響主要體現於相角（Phase, $\theta$）的旋轉。
  - 之後理解單一位元幾何表示法——**布洛赫球面（Bloch Sphere）** 的重要基石。

### 程式實作規劃 (CUDA C)
- 實作複數的基礎代數運算（加、減、乘、除）。
- 實作直角座標與極座標（模數 $r$ 與輻角 $\theta$）的轉換。
- 思考 **AoS (Array of Structures)** 與 **SoA (Structure of Arrays)** 在 GPU 記憶體合併存取（Coalesced Access）上的效能差異。

### run program
```
nvcc -ccbin g++-10 L1_1.cu -o test
```
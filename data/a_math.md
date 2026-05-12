# Chương 1: Nền tảng Đồ thị
##  Ma trận bậc
Ma trận bậc $D$ là một ma trận đường chéo, trong đó phần tử $D_{ii}$ thể hiện tổng số cạnh kết nối với đỉnh $i$. Công thức: $D_{ii} = \sum_{j} A_{ij}$.

# Chương 2: Mạng Nơ-ron Đồ thị (GNN)
## Lan truyền thông tin (Message Passing)
Trong GNN, mỗi đỉnh cập nhật vector đặc trưng của nó bằng cách thu thập thông tin từ các đỉnh lân cận thông qua cơ chế Message Passing.
## Chuẩn hóa đối xứng
Để tránh hiện tượng bùng nổ giá trị đặc trưng đối với các đỉnh có bậc lớn, GCN (Graph Convolutional Network) sử dụng phép chuẩn hóa đối xứng bằng công thức: $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.
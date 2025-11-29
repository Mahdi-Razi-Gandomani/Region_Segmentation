import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class RegionGrowing:
    def __init__(self, img_path, thresh=30, mode="constant"):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Couldn't load image: {img_path}")
        
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.color_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.thresh = thresh
        self.mode = mode
        
        self.seeds = []
        self.regions = []
        self.mask = np.zeros_like(self.gray_img)
        
        # setup display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self._setup_plots()
        
    def _setup_plots(self):
        self.ax1.imshow(self.gray_img, cmap='gray')
        self.ax1.set_title('Click to add seeds (ENTER=done, C=clear)')
        self.ax1.axis('off')
        
        self.ax2.imshow(self.mask, cmap='gray') 
        self.ax2.set_title('Segmentation Result')
        self.ax2.axis('off')
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.tight_layout()
        
    def _grow_from_seed(self, seedPos, threshold=None):
        thresh = threshold if threshold is not None else self.thresh
            
        h, w = self.gray_img.shape
        visited = np.zeros(self.gray_img.shape, dtype=np.uint8)
        q = [seedPos]
        seedVal = float(self.gray_img[seedPos[0], seedPos[1]])
        
        if self.mode == "average":
            pixSum = seedVal
            pixCount = 1
            avgVal = seedVal
        else:
            avgVal = seedVal
        steps = 0
        
        # 4-connected neighbors
        nbrs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        visited[seedPos[0], seedPos[1]] = 255
        
        while q:
            curr = q.pop(0)
            steps += 1
            
            for dy, dx in nbrs:
                ny, nx = curr[0] + dy, curr[1] + dx
                
                if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0:
                    pixVal = float(self.gray_img[ny, nx])
                    diff = abs(pixVal - avgVal)
                    
                    if diff < thresh:
                        visited[ny, nx] = 255
                        q.append((ny, nx))
                        
                        # Update average
                        if self.mode == "average":
                            pixSum += pixVal
                            pixCount += 1
                            avgVal = pixSum / pixCount
        
        return visited, steps
    
    def _merge_masks(self, masks):
        result = np.zeros_like(self.gray_img, dtype=np.float32)
        
        for i, m in enumerate(masks):
            result[m > 0] = (i + 1) * (255 / len(masks))
        
        return result
    

    def _update_display(self):
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.imshow(self.gray_img, cmap='gray')
        
        # draw seed markers
        for idx, pt in enumerate(self.seeds):
            c = Circle((pt[1], pt[0]), 5, color='red', fill=True, alpha=0.7)
            self.ax1.add_patch(c)
            self.ax1.text(pt[1] + 10, pt[0], f'{idx+1}', 
                        color='red', fontsize=12, fontweight='bold')
        
        self.ax1.set_title(f'Seeds: {len(self.seeds)} (Click=add, ENTER=done, C=clear)')
        self.ax1.axis('off')
        
        if len(self.regions) > 0:
            # a combined mask of all regions
            combined_mask = np.zeros_like(self.gray_img, dtype=bool)
            for region_mask in self.regions:
                combined_mask = combined_mask | (region_mask > 0)
            
            # Extract only the segmented parts
            result = np.zeros((self.gray_img.shape[0], self.gray_img.shape[1], 3), dtype=np.uint8)
            result[combined_mask] = self.color_img[combined_mask]
            
            self.ax2.imshow(result)
        else:
            self.ax2.imshow(self.gray_img, cmap='gray')
        
        self.ax2.set_title('Segmentation Result')
        self.ax2.axis('off')
        
        self.fig.canvas.draw()
    
    def _on_click(self, event):
        if event.inaxes == self.ax1 and event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            
            if 0 <= y < self.gray_img.shape[0] and 0 <= x < self.gray_img.shape[1]:
                self.seeds.append((y, x))
                
                reg, stepCount = self._grow_from_seed((y, x))
                self.regions.append(reg)
                
                # print(f"seed at ({y},{x})")
                print(f"Seed #{len(self.seeds)}: pos=({y},{x}), steps={stepCount}")
                
                self._update_display()
    
    def _on_key(self, event):
        if event.key == 'enter':
            print(f"\nDone! Total seeds: {len(self.seeds)}")
            self._show_results()
            
        elif event.key == 'c':
            self.seeds = []
            self.regions = []
            self.mask = np.zeros_like(self.gray_img)
            print("Cleared everything")
            self._update_display()
    
    def _show_results(self):
        plt.close(self.fig)

        if not self.regions:
            print("No regions to show")
            return
        

        
        numRegions = len(self.regions)
        ncols = (numRegions + 1 + 1) // 2
        fig, axes = plt.subplots(2, ncols, figsize=(16, 8))
        axes = axes.flatten()
        
        # individual regions
        for i, region in enumerate(self.regions):
            axes[i].imshow(region, cmap='gray')
            axes[i].set_title(f'Region {i+1}')
            axes[i].axis('off')
        
        # combined overlay 
        combined_mask = np.zeros_like(self.gray_img, dtype=bool)
        for region_mask in self.regions:
            combined_mask = combined_mask | (region_mask > 0)
        
        result = np.zeros((self.gray_img.shape[0], self.gray_img.shape[1], 3), dtype=np.uint8)
        result[combined_mask] = self.color_img[combined_mask]
        
        axes[numRegions].imshow(result)
        axes[numRegions].set_title('Combined')
        axes[numRegions].axis('off')
        
        for i in range(numRegions + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        print("Interative Region Growing\n")
        print("click = add seed point")
        print("ENTER = show results")
        print("C = clear all\n")
        
        plt.show()


if __name__ == "__main__":
    rg = RegionGrowing(
        img_path="input.png",
        thresh=30,
        mode="average")
    rg.run()
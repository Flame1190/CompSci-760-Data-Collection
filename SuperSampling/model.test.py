import unittest
import model.model as mdl
import torch

# TODO: move to a sub-folder without imports breaking
class TestModel(unittest.TestCase):
    def test_zero_upsampling_scale_factor_2(self):
        model = mdl.ZeroUpsampling()
        _input = torch.tensor([[
        [
            [-1,-2],
            [-3,-4]
        ],
        [
            [1,2],
            [3,4]
        ]
        ]], dtype=torch.float32)
        expected = torch.tensor([[
        [
            [-1,0,-2,0],
            [0,0,0,0],
            [-3,0,-4,0],
            [0,0,0,0]
        ],
        [
            [1,0,2,0],
            [0,0,0,0],
            [3,0,4,0],
            [0,0,0,0]
        ]
        ]], dtype=torch.float32)
        self.assertTrue(torch.allclose(model(_input), expected))
    
    def test_zero_upsampling_scale_factor_3(self):
        model = mdl.ZeroUpsampling(scale_factor=3)
        _input = torch.tensor([[
        [
            [-1,-2],
            [-3,-4]
        ],
        [
            [1,2],
            [3,4]
        ]
        ]], dtype=torch.float32)
        expected = torch.tensor([[
        [
            [-1,0,0,-2,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [-3,0,0,-4,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ],
        [
            [1,0,0,2,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [3,0,0,4,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ]
        ]], dtype=torch.float32)
        self.assertTrue(torch.allclose(model(_input), expected))

    def test_backwards_warping(self):
        # this is a bit ehh
        # will have to see how it looks for bigger images
        model = mdl.BackwardsWarping(scale_factor=2)
        upsampler = mdl.ZeroUpsampling(scale_factor=2)

        input_image_unscaled = torch.tensor([[
        [
            [1,2],
            [3,4]
        ]
        ]], dtype=torch.float32)
        input_motion_vectors = torch.tensor([[
        [
            [1,0],
            [1,0]
        ],
        [
            [1,1],
            [0,0]
        ]
        ]], dtype=torch.float32)
        
        # So the warping here might be wrong
        # and instead look like:
        # [4,0,3,0],
        # [0,0,0,0],
        # [2,0,1,0],
        # [0,0,0,0]
        # but will need to visualize
        # The warping and upsampling are both
        # interpolated bilinearly so it should 
        # be okay... (finger's crossed)
        expected_corners = torch.tensor([[
            [
                [4,0,0,3],
                [0,0,0,0],
                [0,0,0,0],
                [2,0,0,1]
            ]
        ]], dtype=torch.float32)

        input_image = upsampler(input_image_unscaled)
        out = model(input_image, input_motion_vectors)

        for i, j in [(0,0), (0,3), (3,0), (3,3)]:
            self.assertTrue(torch.allclose(out[:,:,i,j], expected_corners[:,:,i,j]))
        
    def test_model_pass(self):
        model = mdl.NSRR(num_frames=2, scale_factor=2)
        
        color_maps = [torch.rand(1,3,16,16) * 16 for _ in range(2)]
        depth_maps = [torch.rand(1,1,16,16) for _ in range(2)]
        motion_vectors = [torch.rand(1,2,16,16) * 16]

        out = model(color_maps, depth_maps, motion_vectors)
        self.assertEqual(out.shape, (1,3,32,32))
        out.sum().backward()
        
if __name__ == '__main__':
    unittest.main()
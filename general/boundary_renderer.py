import pygame

class MeshFrame():
    WHITE = (255, 255, 255)
    CYAN = (0,255,255)

    def __init__(self, size):
        super().__init__()
        self.window_name = "Mesh Renderer UI"
        self.size = size
        self.screen = None
        self.create_surface()

    def create_surface(self):
        self.surface = pygame.Surface(self.size)
        self.surface.fill(self.WHITE)

    def render(self):
        screen = pygame.display.set_mode((int(self.size[0]*1.1), int(self.size[1]*1.1)))
        screen.fill(self.WHITE)
        pygame.display.set_caption(self.window_name)

        assert self.surface is not None
        _surface = pygame.transform.flip(self.surface, False, True)

        screen.blit(_surface, (5, 5))
        pygame.display.update()

    def draw_line(self, point1, point2, color=CYAN):
        assert self.surface is not None
        pygame.draw.line(self.surface, color, (point1[0], point1[1]), (point2[0], point2[1]))

    def close(self):
        self.surface = None
        pygame.display.quit()

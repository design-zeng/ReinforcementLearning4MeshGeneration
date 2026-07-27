import pygame


class MeshFrame():
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 255)
    CYAN = (0,255,255)


    def __init__(self, size):
        super().__init__()
        self.window_name = "Mesh Generation UI"  # Name for our window
        self.size = size
        self.screen = None
        self.create_surface()

    def create_surface(self):
        # self.screen = pygame.display.set_mode(self.size)
        # pygame.display.set_caption(self.window_name)
        self.surface = pygame.Surface(self.size)  # Create a Surface to draw on.
        self.surface.fill(self.WHITE)


    def render(self):
        screen = pygame.display.set_mode((int(self.size[0]*1.1), int(self.size[1]*1.1)))
        screen.fill(self.WHITE)
        pygame.display.set_caption(self.window_name)

        # while True:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             quit()
        _surface = pygame.transform.flip(self.surface, False, True)

        screen.blit(_surface, (5, 5))
        pygame.display.update()
        # pygame.display.flip()

    def draw_line(self, point1, point2, color=CYAN):
        pygame.draw.line(self.surface, color, (point1[0], point1[1]), (point2[0], point2[1]))

    def close(self):
        self.surface = None
        self.screen = None
        pygame.display.quit()


# if __name__=="__main__":
#     size = (width, height) = (500, 500)
#     meshF = MeshFrame(size=size)
#     meshF.draw_line((0,0), (50, 50))
#     meshF.render()
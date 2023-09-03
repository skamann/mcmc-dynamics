from ..parameter import Parameters


class Background(object):

    parameters_file = None

    def __init__(self, n_stars: int = None):
        """
        :param n_stars: The number of stars for which the background is modeled.
        """
        self.n_stars = n_stars

    @classmethod
    def default_parameters(cls) -> Parameters:
        """
        :return: The default parameters for this class.
        """
        if cls.parameters_file is None:
            raise NotImplementedError

        return Parameters().load(open(cls.parameters_file))

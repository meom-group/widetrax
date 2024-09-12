# =============================================================================
# filtre_donnees
# =============================================================================

def filtre_donnees(donnees, seuil_min, seuil_max, type_filtre="passe-bas"):
    """
    Filtre les données en fonction des seuils spécifiés et du type de filtre choisi.

    Cette fonction applique un filtre sur les données fournies, soit en mode passe-bas, soit en mode passe-haut.
    Les données en dehors des seuils seront exclues.

    :param donnees: (list ou ndarray) Liste ou tableau des données à filtrer.
    :param seuil_min: (float) Le seuil minimal pour le filtrage des données.
    :param seuil_max: (float) Le seuil maximal pour le filtrage des données.
    :param type_filtre: (str, optionnel) Le type de filtre à appliquer, peut être "passe-bas" ou "passe-haut". Par défaut, "passe-bas".
    
    :return: (list ou ndarray) Les données filtrées.
    
    :raises ValueError: Si le type de filtre n'est ni "passe-bas" ni "passe-haut".
    :raises TypeError: Si le type des données d'entrée n'est pas une liste ou un tableau numpy.
    
    :example:

    >>> donnees = [0.5, 1.5, 2.0, 3.0, 4.5, 5.0]
    >>> filtre_donnees(donnees, seuil_min=1.0, seuil_max=4.0, type_filtre="passe-bas")
    [0.5, 1.5, 2.0, 3.0]
    """
    if type_filtre not in ["passe-bas", "passe-haut"]:
        raise ValueError("Le type de filtre doit être 'passe-bas' ou 'passe-haut'.")
    
    if not isinstance(donnees, (list, np.ndarray)):
        raise TypeError("Les données doivent être une liste ou un tableau numpy.")
    
    if type_filtre == "passe-bas":
        # Garder uniquement les valeurs en dessous du seuil_max
        donnees_filtrees = [x for x in donnees if x <= seuil_max]
    else:  # Passe-haut
        # Garder uniquement les valeurs au-dessus du seuil_min
        donnees_filtrees = [x for x in donnees if x >= seuil_min]
    
    return donnees_filtrees


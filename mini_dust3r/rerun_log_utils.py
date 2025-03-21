from pathlib import Path
import rerun.blueprint as rrb


def create_blueprint(image_name_list: list[Path], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    view_3d = rrb.Spatial3DView(
        origin=f"{log_path}",
        contents=[
            "+ $origin/**",
            *[
                f"- /{log_path}/camera_{i}/pinhole/depth"
                for i in range(len(image_name_list))
            ],
            *[
                f"- /{log_path}/camera_{i}/pinhole/confidence"
                for i in range(len(image_name_list))
            ],
        ],
    )
    if len(image_name_list) > 8:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(contents=view_3d),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    view_3d,
                    rrb.Vertical(
                        contents=[
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(
                                        origin=f"{log_path}/camera_{i}/pinhole/",
                                        contents=[
                                            "+ $origin/**",
                                        ],
                                        name="Pinhole Content",
                                    ),
                                    rrb.Spatial2DView(
                                        origin=f"{log_path}/camera_{i}/pinhole/confidence",
                                        contents=[
                                            "+ $origin/**",
                                        ],
                                        name="Confidence Map",
                                    ),
                                ]
                            )
                            for i in range(len(image_name_list))
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint
